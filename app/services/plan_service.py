import uuid
from typing import Any
from fastapi import HTTPException, status
from langgraph.types import Command

from app.agents.planner import _sanitize_error_message
from app.schemas.request import TravelRequest
from app.schemas.review import HumanReviewInput


def create_plan(graph: Any, request: TravelRequest) -> dict[str, Any]:
    """Start a new travel plan workflow and execute until human review interrupt."""
    plan_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": plan_id}}

    try:
        # Invoke graph up to the interrupt
        graph.invoke({"request": request.model_dump(mode="json")}, config=config)
    except HTTPException:
        raise
    except Exception as exc:
        clean_msg = _sanitize_error_message(str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unable to generate travel plan: {clean_msg}",
        ) from None

    state = graph.get_state(config)

    if state.values.get("status") == "error":
        errors = state.values.get("errors", ["Request validation error"])
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation failed: {'; '.join(errors)}",
        )

    return {
        "plan_id": plan_id,
        "status": state.values.get("status"),
        "workflow_stage": state.values.get("workflow_stage"),
        "draft_itinerary": state.values.get("draft_itinerary"),
        "validation_issues": state.values.get("validation_issues", []),
        "research_results": state.values.get("research_results"),
    }


def get_plan(graph: Any, plan_id: str) -> dict[str, Any]:
    """Retrieve current state of an existing travel plan."""
    config = {"configurable": {"thread_id": plan_id}}
    state = graph.get_state(config)

    if not state.values or "status" not in state.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plan '{plan_id}' not found.",
        )

    return {
        "plan_id": plan_id,
        "status": state.values.get("status"),
        "workflow_stage": state.values.get("workflow_stage"),
        "revision_count": state.values.get("revision_count", 0),
        "draft_itinerary": state.values.get("draft_itinerary"),
        "validation_issues": state.values.get("validation_issues", []),
        "research_results": state.values.get("research_results"),
    }


def review_plan(graph: Any, plan_id: str, review: HumanReviewInput) -> dict[str, Any]:
    """Submit human review decision (approve, reject, or modify) to resume plan."""
    config = {"configurable": {"thread_id": plan_id}}
    state = graph.get_state(config)

    if not state.values or "status" not in state.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plan '{plan_id}' not found.",
        )

    current_status = state.values.get("status")
    if current_status != "awaiting_review":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Plan '{plan_id}' is not awaiting review (current status: '{current_status}').",
        )

    # Resume graph with review payload
    review_payload = review.model_dump(mode="json")
    try:
        graph.invoke(Command(resume=review_payload), config=config)
    except HTTPException:
        raise
    except Exception as exc:
        clean_msg = _sanitize_error_message(str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unable to revise travel plan: {clean_msg}",
        ) from None

    updated_state = graph.get_state(config)
    return {
        "plan_id": plan_id,
        "status": updated_state.values.get("status"),
        "workflow_stage": updated_state.values.get("workflow_stage"),
        "revision_count": updated_state.values.get("revision_count", 0),
        "draft_itinerary": updated_state.values.get("draft_itinerary"),
        "validation_issues": updated_state.values.get("validation_issues", []),
        "research_results": updated_state.values.get("research_results"),
        "final_itinerary": updated_state.values.get("final_itinerary"),
    }


def get_final_plan(graph: Any, plan_id: str) -> dict[str, Any]:
    """Retrieve finalized itinerary for an approved plan."""
    config = {"configurable": {"thread_id": plan_id}}
    state = graph.get_state(config)

    if not state.values or "status" not in state.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plan '{plan_id}' not found.",
        )

    current_status = state.values.get("status")
    final_itinerary = state.values.get("final_itinerary")
    if current_status != "finalized" or not final_itinerary:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Plan '{plan_id}' is not finalized (current status: '{current_status}').",
        )

    return {
        "plan_id": plan_id,
        "status": current_status,
        "final_itinerary": final_itinerary,
        "research_results": state.values.get("research_results"),
    }
