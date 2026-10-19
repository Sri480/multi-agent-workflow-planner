from typing import TypedDict, List, Dict, Any

class WorkflowState(TypedDict, total=False):
    domain: str
    region: str
    research: Dict[str, Any]
    funding: Dict[str, Any]
    pitch_outline: Dict[str, Any]
    feedback: List[Dict[str, Any]]   # replaced each round, not appended
    revision_count: int              # loop guard
    force_review: bool               # demo mode: guarantee one review round
