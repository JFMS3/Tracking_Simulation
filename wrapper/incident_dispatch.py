from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
try:
    from ..environment import ShipEnvironment
except ImportError:
    ShipEnvironment = Any


class Employee(BaseModel):
    employee_id: str
    name: str
    role: str
    skills: list[str]
    available: bool = True


class PositionSnapshot(BaseModel):
    employee_id: str
    position: Optional[tuple[float, float]] = None
    compartment: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class IncidentAnalysis(BaseModel):
    incident_type: str
    severity: Literal["low", "medium", "high", "critical"]
    required_skills: list[str] = Field(default_factory=list)
    preferred_roles: list[str] = Field(default_factory=list)
    recommended_staff_count: int = Field(default=1, ge=1, le=3)
    requires_specialist: bool = False
    safety_notes: list[str] = Field(default_factory=list)


class DispatchDecision(BaseModel):
    selected_employee_ids: list[str] = Field(default_factory=list)
    rationale: str
    alternates: list[str] = Field(default_factory=list)
    escalate: bool = False
    escalation_reason: Optional[str] = None


class DispatchState(TypedDict, total=False):
    incident: str
    incident_location: str
    positions: dict[str, dict[str, Any]]
    employees: list[dict[str, Any]]
    available_skills: list[str]
    available_roles: list[str]
    incident_analysis: dict[str, Any]
    incident_position: Optional[tuple[float, float]]
    resolved_incident_compartment: Optional[str]
    candidates: list[dict[str, Any]]
    decision: dict[str, Any]
    final: dict[str, Any]


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_employees(path: str | Path) -> list[Employee]:
    """Load the deliberately simple pipe-delimited employee store."""
    employees: list[Employee] = []
    path = Path(path)

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 5:
            raise ValueError(
                f"{path}:{line_no}: expected 5 pipe-delimited fields, got {len(parts)}"
            )

        employee_id, name, role, skills_raw, available_raw = parts
        employees.append(
            Employee(
                employee_id=employee_id,
                name=name,
                role=role,
                skills=[s.strip() for s in skills_raw.split(",") if s.strip()],
                available=available_raw.lower() in {"1", "true", "yes", "y"},
            )
        )

    ids = [e.employee_id for e in employees]
    if len(ids) != len(set(ids)):
        raise ValueError("Employee IDs must be unique")
    return employees


class IncidentDispatchGraph:
    """
    LangGraph wrapper around GPT-4o for employee incident dispatch.

    GPT-4o does semantic work:
      1. turn a free-text incident into required skills / preferred roles;
      2. explain/select from a deterministic shortlist.

    Python does factual work:
      - employee availability;
      - skill coverage;
      - room match;
      - geometric distance;
      - localisation-confidence weighting;
      - final output validation.
    """

    def __init__(
        self,
        employee_file: str | Path,
        environment: Optional[ShipEnvironment] = None,
        model: str = "gpt-4o",
        shortlist_size: int = 5,
    ) -> None:
        self.employee_file = Path(employee_file)
        self.environment = environment
        self.shortlist_size = shortlist_size

        base_model = ChatOpenAI(model=model, temperature=0)
        self.incident_model = base_model.with_structured_output(IncidentAnalysis, method="json_schema")
        self.decision_model = base_model.with_structured_output(DispatchDecision, method="json_schema")
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(DispatchState)
        builder.add_node("load_context", self._load_context)
        builder.add_node("analyse_incident", self._analyse_incident)
        builder.add_node("rank_candidates", self._rank_candidates)
        builder.add_node("recommend", self._recommend)
        builder.add_node("validate", self._validate)

        builder.add_edge(START, "load_context")
        builder.add_edge("load_context", "analyse_incident")
        builder.add_edge("analyse_incident", "rank_candidates")
        builder.add_edge("rank_candidates", "recommend")
        builder.add_edge("recommend", "validate")
        builder.add_edge("validate", END)
        return builder.compile()

    def recommend(
        self,
        incident: str,
        incident_location: str,
        current_positions: dict[str, Any],
    ) -> dict[str, Any]:
        """
        current_positions accepts either:

        {
          "EMP001": (3.2, 5.8),
          "EMP002": {"position": (2.0, 1.1), "confidence": 0.85},
          "EMP003": {"compartment": "Room 5", "confidence": 0.75},
        }

        If an environment is supplied, missing compartment names are derived from
        coordinates using environment.compartment_at(position).
        """
        state = self.graph.invoke(
            {
                "incident": incident,
                "incident_location": incident_location,
                "positions": self._normalise_positions(current_positions),
            }
        )
        return state["final"]

    def _normalise_positions(self, current_positions: dict[str, Any]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}

        for employee_id, raw in current_positions.items():
            if raw is None:
                snapshot = PositionSnapshot(employee_id=employee_id, confidence=0.0)
            elif isinstance(raw, (tuple, list)) and len(raw) == 2:
                snapshot = PositionSnapshot(
                    employee_id=employee_id,
                    position=(float(raw[0]), float(raw[1])),
                )
            elif isinstance(raw, dict):
                position = raw.get("position")
                if position is not None:
                    position = (float(position[0]), float(position[1]))
                snapshot = PositionSnapshot(
                    employee_id=employee_id,
                    position=position,
                    compartment=raw.get("compartment"),
                    confidence=float(raw.get("confidence", 1.0)),
                )
            else:
                raise TypeError(f"Unsupported position value for {employee_id}: {raw!r}")

            if (
                snapshot.compartment is None
                and snapshot.position is not None
                and self.environment is not None
            ):
                snapshot.compartment = self.environment.compartment_at(snapshot.position)

            result[employee_id] = snapshot.model_dump()

        return result

    def _resolve_incident_location(
        self, incident_location: str
    ) -> tuple[Optional[str], Optional[tuple[float, float]]]:
        if self.environment is None:
            return incident_location, None

        target = _norm(incident_location)
        exact = [c for c in self.environment.compartments if _norm(c.name) == target]
        if len(exact) == 1:
            comp = exact[0]
            centre = comp.geometry.centroid
            return comp.name, (float(centre.x), float(centre.y))

        # Conservative fuzzy fallback: only accept one unambiguous containment match.
        fuzzy = [
            c for c in self.environment.compartments
            if target and (target in _norm(c.name) or _norm(c.name) in target)
        ]
        if len(fuzzy) == 1:
            comp = fuzzy[0]
            centre = comp.geometry.centroid
            return comp.name, (float(centre.x), float(centre.y))

        return incident_location, None

    def _load_context(self, state: DispatchState) -> dict[str, Any]:
        employees = load_employees(self.employee_file)
        skills = sorted({skill for e in employees for skill in e.skills})
        roles = sorted({e.role for e in employees})
        room, incident_position = self._resolve_incident_location(state["incident_location"])

        return {
            "employees": [e.model_dump() for e in employees],
            "available_skills": skills,
            "available_roles": roles,
            "resolved_incident_compartment": room,
            "incident_position": incident_position,
        }

    def _analyse_incident(self, state: DispatchState) -> dict[str, Any]:
        prompt = f"""
You are an incident-dispatch classifier. Convert the incident into structured
staffing requirements. Do not choose employee names yet.

Incident: {state['incident']}
Location: {state['resolved_incident_compartment']}

The ONLY skill labels you may put in required_skills are drawn from this list:
{state['available_skills']}

Prefer roles only from this list when appropriate:
{state['available_roles']}

Rules:
- required_skills means genuinely required capabilities, not nice-to-haves.
- For an ordinary minor spill, prefer a trained spill-response/housekeeping type
  capability if such a label exists.
- For hazardous, unknown-chemical, fire, medical, electrical, or otherwise
  dangerous incidents, mark severity/requires_specialist conservatively and add
  a short safety note.
- Do not invent certifications or facts about employees.
- recommended_staff_count should normally be 1, and only increase when the
  incident itself justifies multiple responders.
""".strip()

        analysis = self.incident_model.invoke(prompt)
        # Drop any unexpected labels even if a provider/model ignores the prompt.
        allowed = set(state["available_skills"])
        analysis.required_skills = [s for s in analysis.required_skills if s in allowed]
        return {"incident_analysis": analysis.model_dump()}

    def _rank_candidates(self, state: DispatchState) -> dict[str, Any]:
        analysis = IncidentAnalysis.model_validate(state["incident_analysis"])
        required = set(analysis.required_skills)
        preferred_roles = {_norm(r) for r in analysis.preferred_roles}
        incident_compartment = state.get("resolved_incident_compartment")
        incident_position = state.get("incident_position")
        position_map = state["positions"]

        candidates: list[dict[str, Any]] = []
        for raw_employee in state["employees"]:
            employee = Employee.model_validate(raw_employee)
            if not employee.available:
                continue

            pos = PositionSnapshot.model_validate(
                position_map.get(employee.employee_id, {"employee_id": employee.employee_id, "confidence": 0.0})
            )
            employee_skills = set(employee.skills)
            matched_skills = sorted(required & employee_skills)
            missing_skills = sorted(required - employee_skills)
            skill_coverage = (len(matched_skills) / len(required)) if required else 1.0

            # Required skills are a hard gate. Preferred roles remain a soft ranking signal.
            qualified = not missing_skills

            same_compartment = bool(
                incident_compartment
                and pos.compartment
                and _norm(incident_compartment) == _norm(pos.compartment)
            )

            distance_m: Optional[float] = None
            if incident_position is not None and pos.position is not None:
                distance_m = math.dist(incident_position, pos.position)

            skill_score = 60.0 * skill_coverage
            role_score = 10.0 if _norm(employee.role) in preferred_roles else 0.0
            room_score = 20.0 if same_compartment else 0.0
            distance_score = 0.0 if distance_m is None else 15.0 / (1.0 + distance_m / 5.0)
            confidence_score = 5.0 * pos.confidence
            total = skill_score + role_score + room_score + distance_score + confidence_score

            candidates.append(
                {
                    "employee_id": employee.employee_id,
                    "name": employee.name,
                    "role": employee.role,
                    "skills": employee.skills,
                    "matched_skills": matched_skills,
                    "missing_skills": missing_skills,
                    "qualified": qualified,
                    "position": pos.position,
                    "compartment": pos.compartment,
                    "distance_m": None if distance_m is None else round(distance_m, 2),
                    "same_compartment": same_compartment,
                    "position_confidence": pos.confidence,
                    "score": round(total, 2),
                }
            )

        candidates.sort(key=lambda c: (c["qualified"], c["score"]), reverse=True)
        return {"candidates": candidates[: self.shortlist_size]}

    def _recommend(self, state: DispatchState) -> dict[str, Any]:
        analysis = IncidentAnalysis.model_validate(state["incident_analysis"])
        eligible = [c for c in state["candidates"] if c["qualified"]]

        if not eligible:
            return {
                "decision": DispatchDecision(
                    selected_employee_ids=[],
                    rationale="No available employee in the shortlist satisfies the specialist requirements.",
                    alternates=[],
                    escalate=True,
                    escalation_reason="No qualified available responder was found.",
                ).model_dump()
            }

        prompt = f"""
You are choosing responders from a pre-scored shortlist. You may ONLY select
employee IDs that appear below. Never invent an employee or capability.

Incident: {state['incident']}
Location: {state['resolved_incident_compartment']}
Incident analysis: {analysis.model_dump()}
Shortlisted candidates: {eligible}

Selection rules:
- Qualification and required-skill coverage come first.
- Then prefer suitable role and proximity.
- Use the supplied score as a strong ranking signal, not as a fact to override
  missing required skills.
- Select approximately {analysis.recommended_staff_count} employee(s).
- Keep the rationale concise and evidence-based using only the supplied fields.
- If the incident is critical/high-risk and this shortlist is inadequate, set
  escalate=true rather than pretending it is safe.
""".strip()

        decision = self.decision_model.invoke(prompt)
        return {"decision": decision.model_dump()}

    def _validate(self, state: DispatchState) -> dict[str, Any]:
        analysis = IncidentAnalysis.model_validate(state["incident_analysis"])
        decision = DispatchDecision.model_validate(state["decision"])
        candidates = [c for c in state["candidates"] if c["qualified"]]
        candidate_ids = {c["employee_id"] for c in candidates}

        # Remove hallucinated IDs and duplicates while preserving model order.
        selected: list[str] = []
        for employee_id in decision.selected_employee_ids:
            if employee_id in candidate_ids and employee_id not in selected:
                selected.append(employee_id)

        needed = min(analysis.recommended_staff_count, len(candidates))
        if not decision.escalate and len(selected) < needed:
            for candidate in candidates:
                employee_id = candidate["employee_id"]
                if employee_id not in selected:
                    selected.append(employee_id)
                if len(selected) >= needed:
                    break

        # If a specialist is required but none survived validation, force escalation.
        if analysis.requires_specialist and not selected:
            decision.escalate = True
            decision.escalation_reason = decision.escalation_reason or "No qualified specialist is available."

        decision.selected_employee_ids = selected
        decision.alternates = [
            employee_id
            for employee_id in decision.alternates
            if employee_id in candidate_ids and employee_id not in selected
        ]

        selected_details = [c for c in candidates if c["employee_id"] in selected]

        return {
            "final": {
                "incident": state["incident"],
                "location": state["resolved_incident_compartment"],
                "analysis": analysis.model_dump(),
                "recommended_employee_ids": selected,
                "recommended_employees": selected_details,
                "rationale": decision.rationale,
                "alternates": decision.alternates,
                "escalate": decision.escalate,
                "escalation_reason": decision.escalation_reason,
                "shortlist": state["candidates"],
            }
        }
