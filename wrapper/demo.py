from incident_dispatch import IncidentDispatchGraph
from pathlib import Path
from dotenv import load_dotenv


WRAPPER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WRAPPER_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")


# In real program, would pass environment=<ShipEnvironment instance> 
dispatcher = IncidentDispatchGraph(
    employee_file=WRAPPER_DIR  / "employees.txt",
    environment=None,
    model="gpt-4o",
)
dispatcher.graph.get_graph().draw_mermaid_png(
    output_file_path="dispatch_graph.png"
)

current_positions = {
    "EMP001": {"position": (6.1, 4.7), "compartment": "Room 4", "confidence": 0.90},
    "EMP002": {"position": (7.0, 4.8), "compartment": "Room 5", "confidence": 0.84},
    "EMP003": {"position": (6.8, 4.4), "compartment": "Room 5", "confidence": 0.78},
    "EMP004": {"position": (2.1, 1.2), "compartment": "Engine Room", "confidence": 0.91},
    "EMP005": {"position": (3.0, 5.9), "compartment": "Medical", "confidence": 0.88},
    "EMP006": {"position": (4.0, 3.0), "compartment": "Corridor", "confidence": 0.80},
}

result = dispatcher.recommend(
    incident="There is a small water spill on the floor.",
    incident_location="Room 5",
    current_positions=current_positions,
)

print("Recommended:", result["recommended_employee_ids"])
print("Reason:", result["rationale"])
print("Escalate:", result["escalate"])