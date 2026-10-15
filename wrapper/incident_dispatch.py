from __future__ import annotations

import math
import re
import sqlite3
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
    """What GPT-4o returns from step 2 (analyse_incident)."""
    incident_type: str
    severity: Literal["low", "medium", "high", "critical"]
    required_skills: list[str] = Field(default_factory=list)
    preferred_roles: list[str] = Field(default_factory=list)
    recommended_staff_count: int = Field(default=1, ge=1, le=3)
    requires_specialist: bool = False
    safety_notes: list[str] = Field(default_factory=list)


class DispatchDecision(BaseModel):
    """What GPT-4o returns from step 4 (recommend)."""
    selected_employee_ids: list[str] = Field(default_factory=list)
    rationale: str
    alternates: list[str] = Field(default_factory=list)
    escalate: bool = False
    escalation_reason: Optional[str] = None


class DispatchState(TypedDict, total=False):
    """Shared state passed between graph steps. Each step adds some keys."""
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
    proposed_team: list[str]          # IDs that together cover the required skills
    uncovered_skills: list[str]       # required skills no available employee has
    decision: dict[str, Any]
    final: dict[str, Any]


# Helper functions 
def _norm(value: str) -> str:
    """Lowercase and strip punctuation, so "Fire Response" == "fire_response"."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_employees(db_path: str | Path) -> tuple[list[Employee], list[str]]:
    """
    Load employees and the full skill list from the SQLite database.

    Returns (employees, all skill names in the `skills` table).
    """
    db_path = Path(db_path)
    if not db_path.exists():
        # sqlite3.connect would otherwise silently create an empty database.
        raise FileNotFoundError(f"Employee database not found: {db_path}")

    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT e.id, e.forename, e.surname, e.role, e.available, s.name
            FROM employees AS e
            LEFT JOIN employee_skills_map AS m ON m.employee_id = e.id
            LEFT JOIN skills AS s ON s.skill_id = m.skill_id
            ORDER BY e.id
            """
        ).fetchall()
        all_skills = [name for (name,) in con.execute("SELECT name FROM skills")]
    finally:
        con.close()

    # The join gives one row per (employee, skill); group them per employee.
    employees: dict[str, Employee] = {}
    for emp_id, forename, surname, role, available, skill in rows:
        emp_id = str(emp_id)
        if emp_id not in employees:
            employees[emp_id] = Employee(
                employee_id=emp_id,
                name=f"{forename or ''} {surname or ''}".strip(),
                role=role or "",
                skills=[],
                # SQLite has no real boolean: the column holds text like
                # 'true'/'false' (and bool('false') is True in Python).
                available=str(available).strip().lower() in {"1", "true", "yes"},
            )
        if skill:
            employees[emp_id].skills.append(skill)

    return list(employees.values()), sorted(all_skills)


def match_skill_names(requested: list[str], known: list[str]) -> list[str]:
    """
    Replace each skill name the model gave with the matching database label
    (ignoring case and punctuation). Names with no match are kept as-is, so
    they show up later as uncovered skills instead of silently disappearing.
    """
    by_norm = {_norm(k): k for k in known}
    result: list[str] = []
    for name in requested:
        label = by_norm.get(_norm(name), name)
        if label not in result:
            result.append(label)
    return result


def pick_team(
    candidates: list[dict[str, Any]],
    required: list[str],
    already_selected: Optional[list[str]] = None,
) -> tuple[list[str], list[str]]:
    """
    Greedily pick employees until together they hold every required skill.

    Each round adds whoever covers the most still-missing skills (ties go to
    the higher score). If one person has everything, they are picked alone.

    Returns (IDs added, required skills nobody in `candidates` has).
    """
    already_selected = already_selected or []
    missing = set(required)
    for c in candidates:
        if c["employee_id"] in already_selected:
            missing -= set(c["matched_skills"])

    added: list[str] = []
    pool = [c for c in candidates if c["employee_id"] not in already_selected]
    while missing and pool:
        best = max(pool, key=lambda c: (len(missing & set(c["matched_skills"])), c["score"]))
        if not missing & set(best["matched_skills"]):
            break  # nobody left can help
        added.append(best["employee_id"])
        missing -= set(best["matched_skills"])
        pool.remove(best)

    return added, sorted(missing)


# Dispatch graph
class IncidentDispatchGraph:
    """
    LangGraph pipeline: load_context -> analyse_incident -> rank_candidates
                        -> recommend -> validate
    """

    def __init__(
        self,
        employee_db: str | Path,
        environment: Optional[ShipEnvironment] = None,
        model: str = "gpt-4o",
        shortlist_size: int = 5,
    ) -> None:
        self.employee_db = Path(employee_db)
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
        current_positions: dict[Any, Any],
    ) -> dict[str, Any]:
        """
        Run the full pipeline and return the final recommendation dict.

        current_positions is keyed by employee ID (1 or "1" both work):

        {
          1: (3.2, 5.8),
          2: {"position": (2.0, 1.1), "confidence": 0.85},
          3: {"compartment": "Laundry", "confidence": 0.75},
        }

        If an environment is supplied, missing compartment names are derived
        from coordinates using environment.compartment_at(position).
        """
        state = self.graph.invoke(
            {
                "incident": incident,
                "incident_location": incident_location,
                "positions": self._normalise_positions(current_positions),
            }
        )
        return state["final"]

    def _normalise_positions(self, current_positions: dict[Any, Any]) -> dict[str, dict[str, Any]]:
        """Convert the accepted position formats into PositionSnapshot dicts."""
        result: dict[str, dict[str, Any]] = {}

        for raw_id, raw in current_positions.items():
            employee_id = str(raw_id)
            if raw is None:
                snapshot = PositionSnapshot(employee_id=employee_id, confidence=0.0)
            elif isinstance(raw, (tuple, list)):
                snapshot = PositionSnapshot(employee_id=employee_id, position=tuple(raw))
            else:
                snapshot = PositionSnapshot(employee_id=employee_id, **raw)

            if snapshot.compartment is None and snapshot.position is not None and self.environment:
                snapshot.compartment = self.environment.compartment_at(snapshot.position)

            result[employee_id] = snapshot.model_dump()

        return result

    def _resolve_incident_location(
        self, incident_location: str
    ) -> tuple[Optional[str], Optional[tuple[float, float]]]:
        """Match the location text to a compartment; return (name, centroid)."""
        if self.environment is None:
            return incident_location, None

        target = _norm(incident_location)
        exact = [c for c in self.environment.compartments if _norm(c.name) == target]
        # Fallback: accept a partial name match only if it is unambiguous.
        partial = [
            c for c in self.environment.compartments
            if target and (target in _norm(c.name) or _norm(c.name) in target)
        ]
        for matches in (exact, partial):
            if len(matches) == 1:
                centre = matches[0].geometry.centroid
                return matches[0].name, (float(centre.x), float(centre.y))

        return incident_location, None


    def _load_context(self, state: DispatchState) -> dict[str, Any]:
        employees, all_skills = load_employees(self.employee_db)
        room, incident_position = self._resolve_incident_location(state["incident_location"])

        return {
            "employees": [e.model_dump() for e in employees],
            "available_skills": all_skills,
            "available_roles": sorted({e.role for e in employees}),
            "resolved_incident_compartment": room,
            "incident_position": incident_position,
        }

    def _analyse_incident(self, state: DispatchState) -> dict[str, Any]:
        prompt = f"""
        You are an incident-dispatch classifier. Convert the incident into structured
        staffing requirements. Do not choose employee names yet.

        Incident: {state['incident']}
        Location: {state['resolved_incident_compartment']}

        Skill labels used by the crew database:
        {state['available_skills']}

        Roles on board:
        {state['available_roles']}

        Rules:
        - required_skills means genuinely required capabilities, not nice-to-haves.
        - Use the exact label from the skill list whenever one fits. If a required
        capability has no matching label, still include it as a short plain name.
        Never leave out a required capability or swap in an unrelated label.
        - Skills may be split across responders: list everything the incident needs,
        even if no single person would have all of it.
        - For an ordinary minor spill, prefer a spill-response/housekeeping skill.
        - For hazardous, unknown-chemical, fire, medical, electrical, or otherwise
        dangerous incidents, mark severity/requires_specialist conservatively and
        add a short safety note.
        - recommended_staff_count should normally be 1, and only increase when the
        incident itself justifies multiple responders.
        """.strip()

        analysis = self.incident_model.invoke(prompt)
        analysis.required_skills = match_skill_names(analysis.required_skills, state["available_skills"])
        return {"incident_analysis": analysis.model_dump()}

    def _rank_candidates(self, state: DispatchState) -> dict[str, Any]:
        analysis = IncidentAnalysis.model_validate(state["incident_analysis"])
        required = analysis.required_skills
        preferred_roles = {_norm(r) for r in analysis.preferred_roles}
        incident_compartment = state.get("resolved_incident_compartment")
        incident_position = state.get("incident_position")

        candidates: list[dict[str, Any]] = []
        for raw_employee in state["employees"]:
            employee = Employee.model_validate(raw_employee)
            if not employee.available:
                continue

            pos = PositionSnapshot.model_validate(
                state["positions"].get(employee.employee_id)
                or {"employee_id": employee.employee_id, "confidence": 0.0}
            )

            their_skills = {_norm(s) for s in employee.skills}
            matched_skills = [s for s in required if _norm(s) in their_skills]
            skill_coverage = len(matched_skills) / len(required) if required else 1.0

            same_compartment = bool(
                incident_compartment
                and pos.compartment
                and _norm(incident_compartment) == _norm(pos.compartment)
            )
            distance_m = (
                math.dist(incident_position, pos.position)
                if incident_position is not None and pos.position is not None
                else None
            )

            score = (
                60.0 * skill_coverage
                + (10.0 if _norm(employee.role) in preferred_roles else 0.0)
                + (20.0 if same_compartment else 0.0)
                + (15.0 / (1.0 + distance_m / 5.0) if distance_m is not None else 0.0)
                + 5.0 * pos.confidence
            )

            candidates.append(
                {
                    "employee_id": employee.employee_id,
                    "name": employee.name,
                    "role": employee.role,
                    "skills": employee.skills,
                    "matched_skills": matched_skills,
                    "missing_skills": [s for s in required if s not in matched_skills],
                    # Has every required skill on their own.
                    "qualified": len(matched_skills) == len(required),
                    # Has at least one required skill, so can be part of a team.
                    "contributes": bool(matched_skills) or not required,
                    "position": pos.position,
                    "compartment": pos.compartment,
                    "distance_m": None if distance_m is None else round(distance_m, 2),
                    "same_compartment": same_compartment,
                    "position_confidence": pos.confidence,
                    "score": round(score, 2),
                }
            )

        candidates.sort(key=lambda c: (c["qualified"], c["contributes"], c["score"]), reverse=True)

        # Choose a covering team from everyone, then make sure those people
        # are in the shortlist even if they rank below the cut-off.
        team, uncovered = pick_team([c for c in candidates if c["contributes"]], required)
        shortlist = candidates[: self.shortlist_size]
        shortlist += [c for c in candidates if c["employee_id"] in team and c not in shortlist]

        return {"candidates": shortlist, "proposed_team": team, "uncovered_skills": uncovered}


    def _recommend(self, state: DispatchState) -> dict[str, Any]:
        analysis = IncidentAnalysis.model_validate(state["incident_analysis"])
        eligible = [c for c in state["candidates"] if c["contributes"]]
        team = state["proposed_team"]

        if not eligible:
            # Nobody available has any required skill: no point asking the LLM.
            return {
                "decision": DispatchDecision(
                    rationale="No available employee has any of the required skills.",
                    escalate=True,
                    escalation_reason="No suitable responder available.",
                ).model_dump()
            }

        prompt = f"""
        You are choosing responders from a pre-scored shortlist. You may ONLY select
        employee IDs that appear below. Never invent an employee or capability.

        Incident: {state['incident']}
        Location: {state['resolved_incident_compartment']}
        Incident analysis: {analysis.model_dump()}
        Required skills NO available employee has: {state['uncovered_skills'] or "none"}
        Proposed team (together covers the required skills): {team}
        Shortlisted candidates: {eligible}

        Selection rules:
        - The selected GROUP must together cover every required skill. One person
        does not need all of them; "matched_skills" shows what each person covers.
        - Start from the proposed team. Change it only for a clear reason visible in
        the supplied fields (e.g. better role fit or proximity at equal coverage).
        - Select approximately {max(analysis.recommended_staff_count, len(team))} employee(s).
        - Keep the rationale concise, and say who covers which skill.
        - If the incident is high-risk and this shortlist is inadequate, or a required
        skill is not covered by anyone, set escalate=true.
        """.strip()

        decision = self.decision_model.invoke(prompt)
        return {"decision": decision.model_dump()}

    # ---- Step 5 ----------------------------------------------------------
    def _validate(self, state: DispatchState) -> dict[str, Any]:
        """Check the LLM's choice against the facts and fix it if needed."""
        analysis = IncidentAnalysis.model_validate(state["incident_analysis"])
        decision = DispatchDecision.model_validate(state["decision"])
        candidates = [c for c in state["candidates"] if c["contributes"]]
        by_id = {c["employee_id"]: c for c in candidates}
        uncovered = state["uncovered_skills"]

        # 1. Keep only real shortlisted IDs (drops hallucinations and duplicates).
        selected = list(dict.fromkeys(e for e in decision.selected_employee_ids if e in by_id))

        # 2. Add people until the group covers every required skill it can.
        added, _ = pick_team(candidates, analysis.required_skills, selected)
        selected += added

        # 3. Top up to the recommended head count with the best-ranked people.
        needed = min(analysis.recommended_staff_count, len(candidates))
        for c in candidates:
            if len(selected) >= needed:
                break
            if c["employee_id"] not in selected:
                selected.append(c["employee_id"])

        # 4. Escalate if a skill is missing or nobody could be sent
        if uncovered:
            decision.escalate = True
            decision.escalation_reason = "No available employee has: " + ", ".join(uncovered)
        elif not selected:
            decision.escalate = True
            decision.escalation_reason = decision.escalation_reason or "No suitable responder available."

        return {
            "final": {
                "incident": state["incident"],
                "location": state["resolved_incident_compartment"],
                "analysis": analysis.model_dump(),
                "recommended_employee_ids": selected,
                "recommended_employees": [by_id[e] for e in selected],
                "skill_coverage": {
                    skill: [e for e in selected if skill in by_id[e]["matched_skills"]]
                    for skill in analysis.required_skills
                },
                "rationale": decision.rationale,
                "alternates": [e for e in decision.alternates if e in by_id and e not in selected],
                "escalate": decision.escalate,
                "escalation_reason": decision.escalation_reason,
                "uncovered_skills": uncovered,
                "shortlist": state["candidates"],
            }
        }