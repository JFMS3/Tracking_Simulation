# Incident dispatch

`incident_dispatch.py` recommends which crew member(s) to send to an incident on the ship.

You give it three things:

- a free-text description of the incident ("There is a pipe leakage and a fire!");
- where the incident is;
- where the crew currently are.

It returns who to send, why, and whether a human needs to step in.

The work is split by what each side is good at:

| GPT-4o (judgement) | Python (facts) |
|---|---|
| Works out which skills the incident needs | Who is available |
| Picks responders from a shortlist and explains why | Who has which skills |
| Decides whether the situation looks unsafe | Who is in or near the incident compartment |
| | Scoring, and checking that the LLM's choice is valid |

The LLM never decides facts. Every ID it returns is checked, and the final team is guaranteed to cover the required skills wherever the crew makes that possible.

---

## Quick start

```python
from environment import ShipEnvironment
from incident_dispatch import IncidentDispatchGraph

env = ShipEnvironment.from_config("ship.yaml")          # optional
dispatcher = IncidentDispatchGraph("Employees.db", environment=env)

result = dispatcher.recommend(
    incident="There is a pipe leakage and a fire!",
    incident_location="Laundry",
    current_positions={
        1: (3.2, 5.8),                                   # (x, y), confidence 1.0
        6: {"position": (2.0, 1.1), "confidence": 0.85},
        7: {"compartment": "Laundry", "confidence": 0.75},
    },
)

print(result["recommended_employee_ids"], result["rationale"])
if result["escalate"]:
    print("ESCALATE:", result["escalation_reason"])
```

`OPENAI_API_KEY` must be set in the environment.

**The environment is optional.** Without it:

- the incident location is used as plain text;
- distance isn't scored;
- employees only get the same-compartment bonus if you give their `compartment` directly.

**Employees missing from `current_positions`** are still considered. They just get no location score.

---

## Employee database

The SQLite file needs these three tables:

```
employees(id, forename, surname, role, available)
skills(skill_id, name)
employee_skills_map(employee_id, skill_id)     -- one row per (employee, skill)
```

- `available` can be `'true'`/`'false'` text or `1`/`0`.
- Employee IDs come back as strings (`"1"`, `"2"`, …). In `current_positions` you can use `1` or `"1"`.
- The database is re-read on every call, so changes take effect immediately.
- The model is shown **all** skills in the `skills` table, including skills only unavailable staff have. That way an incident that needs one of those skills is reported as a gap and escalated, rather than ignored.

---

## How it works

The pipeline is a LangGraph graph of five steps in a straight line. They share one state dict, and each step adds keys to it.

```
load_context → analyse_incident → rank_candidates → recommend → validate
```

### 1. `load_context`
- Loads employees and skills from the database.
- Matches the location text to a compartment: an exact name match first, otherwise a partial match, but only if exactly one compartment fits. It uses that compartment's centre point as the incident position.

### 2. `analyse_incident` (GPT-4o)
- Turns the incident text into an `IncidentAnalysis`: `required_skills`, `preferred_roles`, `severity`, `recommended_staff_count` (1–3), `requires_specialist` and `safety_notes`.
- `match_skill_names` then maps each skill name onto the database label, ignoring case and punctuation, so `"Fire Response"` becomes `fire_response`.
- A skill the model asks for that isn't in the database is **kept**, not dropped. It will show up later as uncovered and cause an escalation.

### 3. `rank_candidates` (Python)
Scores every available employee:

| Component | Points |
|---|---|
| Required-skill coverage | 60 × fraction of required skills they have |
| Role in `preferred_roles` | +10 |
| In the incident compartment | +20 |
| Distance to incident | 15 / (1 + d/5), e.g. 15 at 0 m, 7.5 at 5 m |
| Localisation confidence | 5 × confidence |

Each candidate also gets two flags:

- `qualified`: has *every* required skill on their own.
- `contributes`: has *at least one* required skill, so can be part of a team.

`pick_team` then builds a team that **together** covers the required skills. It does this greedily: at each step it adds whoever covers the most still-missing skills, and uses score to break ties. If one person has everything, they're picked alone. Required skills that nobody available has are recorded as `uncovered_skills`.

The shortlist is the top `shortlist_size` candidates (default 5), plus anyone in the proposed team who fell below that cut.

### 4. `recommend` (GPT-4o)
The model sees the shortlisted candidates who contribute, the proposed team, and any uncovered skills. It returns a `DispatchDecision`: who to send, a rationale, alternates, and whether to escalate.

If nobody available has any required skill, this step skips the LLM and escalates straight away.

### 5. `validate` (Python)
This step corrects the LLM's answer where it doesn't fit the facts:

1. Removes IDs that aren't in the shortlist (made-up or duplicated).
2. Adds people until the group covers every required skill that someone available has.
3. Adds the best-ranked remaining people until it reaches `recommended_staff_count`.
4. Forces escalation if any required skill is uncovered, or if nobody could be sent.

---

## Escalation

`escalate` is **only a flag in the output**. This module doesn't notify anyone. Whatever calls `recommend()` should check it and alert a human, such as the officer of the watch.

It's set when:

- a required skill isn't held by any available employee, including skills not in the database at all (enforced by Python);
- nobody suitable could be selected (enforced by Python);
- GPT-4o judges the incident too risky for the shortlist (the model's call).

When a skill is uncovered, the module still recommends the best people for the parts that *can* be covered. For example, for a leak and a fire it sends someone for the leak and escalates for the fire.

---

## Output

`recommend()` returns a dict with these keys:

| Key | Meaning |
|---|---|
| `recommended_employee_ids` | Who to send |
| `recommended_employees` | Their full candidate records (skills, location, score) |
| `skill_coverage` | `{required skill: [IDs covering it]}`; an empty list means uncovered |
| `rationale` | GPT-4o's explanation |
| `alternates` | Backup options suggested by GPT-4o (validated) |
| `escalate`, `escalation_reason` | See above |
| `uncovered_skills` | Required skills no available employee has |
| `analysis` | The full `IncidentAnalysis` from step 2 |
| `shortlist` | All scored candidates the LLM could choose from, useful for debugging |
| `incident`, `location` | Echo of the input, with the location resolved to a compartment name |

---

## Known limitations

- **Distance is a straight line.** It ignores walls and doors, so someone close but behind a bulkhead can outrank someone further away down an open corridor.
- **Confidence doesn't scale proximity.** A low-confidence position still gets the full room and distance bonus. Confidence only adds up to 5 points on its own.
- **Localiser output needs converting.** The localisers in `positioning.py` return a covariance matrix, not a 0–1 confidence, so you need to convert one to the other. Bare `(x, y)` tuples default to confidence 1.0.
- **The LLM can over-select.** If it picks more people than recommended, they are all kept. The count is never trimmed.
- **The team is chosen greedily.** It's near-optimal and fine for a crew of this size, but not guaranteed to be the smallest possible team.
