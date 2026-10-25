from typing import Any, Optional, TypedDict


class TravelPlanState(TypedDict, total=False):
    request: dict[str, Any]

    research_results: Optional[dict[str, Any]]
    draft_itinerary: Optional[dict[str, Any]]
    validation_issues: list[str]

    review_action: Optional[str]
    review_feedback: Optional[str]
    requested_modifications: Optional[str]

    revision_count: int
    status: str
    workflow_stage: str

    final_itinerary: Optional[dict[str, Any]]
    errors: list[str]
