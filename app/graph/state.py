"""LangGraph state definition for the AI Travel Planner workflow."""

from typing import Any, Optional, TypedDict


class TravelPlanState(TypedDict, total=False):
    """Workflow state for travel planning and validation."""

    # Original travel request payload (serializable dict)
    request: dict[str, Any]

    # Agent outputs
    research_results: Optional[dict[str, Any]]
    draft_itinerary: Optional[dict[str, Any]]
    validation_issues: list[str]

    # Human review action and instructions
    review_action: Optional[str]
    review_feedback: Optional[str]
    requested_modifications: Optional[str]

    # Execution tracking and status
    revision_count: int
    status: str
    workflow_stage: str

    # Final outputs and errors
    final_itinerary: Optional[dict[str, Any]]
    errors: list[str]
