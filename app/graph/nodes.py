from typing import Any
from langgraph.types import interrupt
from app.agents.itinerary_validator import run_itinerary_validator
from app.agents.planner import run_planner_agent
from app.agents.researcher import run_researcher_agent
from app.graph.state import TravelPlanState
from app.schemas.request import TravelRequest
from app.schemas.review import HumanReviewInput


def validate_request(state: TravelPlanState) -> dict[str, Any]:
    raw_request = state.get("request", {})
    try:
        validated = TravelRequest.model_validate(raw_request)
        return {
            "request": validated.model_dump(mode="json"),
            "status": "validated",
            "workflow_stage": "validation",
            "errors": [],
            "revision_count": state.get("revision_count", 0),
        }
    except Exception as exc:
        return {
            "status": "error",
            "workflow_stage": "validation",
            "errors": [str(exc)],
        }


def research_agent(state: TravelPlanState) -> dict[str, Any]:
    request = state["request"]
    research = run_researcher_agent(request)
    return {
        "research_results": research,
        "status": "researched",
        "workflow_stage": "research",
    }


def planner_agent(state: TravelPlanState) -> dict[str, Any]:
    request = state["request"]
    research = state.get("research_results")
    draft = run_planner_agent(
        request=request,
        research=research,
        revision_count=state.get("revision_count", 0),
    )
    return {
        "draft_itinerary": draft,
        "status": "planning_completed",
        "workflow_stage": "planning",
    }


def itinerary_validator(state: TravelPlanState) -> dict[str, Any]:
    request = state["request"]
    draft = state.get("draft_itinerary")
    research = state.get("research_results")
    issues = run_itinerary_validator(
        request=request,
        draft_itinerary=draft,
        research=research,
    )
    return {
        "validation_issues": issues,
        "status": "validated_itinerary",
        "workflow_stage": "validation",
    }


def mark_awaiting_review(state: TravelPlanState) -> dict[str, Any]:
    return {
        "status": "awaiting_review",
        "workflow_stage": "awaiting_review",
    }


def human_review(state: TravelPlanState) -> dict[str, Any]:
    # Halt graph execution here; resumed later with Command(resume=review_payload)
    review_data = interrupt({
        "message": "Itinerary ready for review",
        "draft_itinerary": state.get("draft_itinerary"),
        "validation_issues": state.get("validation_issues", []),
        "revision_count": state.get("revision_count", 0),
    })

    review = HumanReviewInput.model_validate(review_data)
    action_str = review.action.value if hasattr(review.action, "value") else str(review.action)

    return {
        "review_action": action_str,
        "review_feedback": review.feedback,
        "requested_modifications": review.modifications,
        "status": f"reviewed_{action_str}",
        "workflow_stage": "human_review",
    }


def revise_itinerary(state: TravelPlanState) -> dict[str, Any]:
    current_revisions = state.get("revision_count", 0)
    new_revisions = current_revisions + 1
    request = state["request"]
    research = state.get("research_results")
    feedback = state.get("review_feedback")
    modifications = state.get("requested_modifications")

    revised_draft = run_planner_agent(
        request=request,
        research=research,
        feedback=feedback,
        modifications=modifications,
        revision_count=new_revisions,
    )
    return {
        "draft_itinerary": revised_draft,
        "revision_count": new_revisions,
        "status": "itinerary_revised",
        "workflow_stage": "revision",
    }


def finalize_plan(state: TravelPlanState) -> dict[str, Any]:
    return {
        "final_itinerary": state.get("draft_itinerary"),
        "status": "finalized",
        "workflow_stage": "completed",
    }


def mark_revision_limit(state: TravelPlanState) -> dict[str, Any]:
    return {
        "status": "revision_limit_reached",
        "workflow_stage": "ended_revision_limit",
    }
