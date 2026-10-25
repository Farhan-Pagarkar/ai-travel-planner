from typing import Literal, Optional
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from app.graph.nodes import (
    finalize_plan,
    human_review,
    itinerary_validator,
    mark_awaiting_review,
    mark_revision_limit,
    planner_agent,
    research_agent,
    revise_itinerary,
    validate_request,
)
from app.graph.state import TravelPlanState

MAX_REVISIONS = 3


def route_request_validation(state: TravelPlanState) -> Literal["research_agent", "end"]:
    #Route after request validation: proceed if valid, terminate on error.
    if state.get("errors"):
        return "end"
    return "research_agent"


def route_after_human_review(
    state: TravelPlanState,
) -> Literal["finalize_plan", "revise_itinerary", "mark_revision_limit"]:
    #Route after human review: finalize on approve, revise on reject/modify up to limit.
    action = state.get("review_action")
    if action == "approve":
        return "finalize_plan"

    revisions = state.get("revision_count", 0)
    if revisions >= MAX_REVISIONS:
        return "mark_revision_limit"
    return "revise_itinerary"


def create_travel_planner_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    # Build and compile the travel planner StateGraph.
    # The checkpointer lifecycle is managed by the caller.
    builder = StateGraph(TravelPlanState)

    # 1. Register nodes
    builder.add_node("validate_request", validate_request)
    builder.add_node("research_agent", research_agent)
    builder.add_node("planner_agent", planner_agent)
    builder.add_node("itinerary_validator", itinerary_validator)
    builder.add_node("mark_awaiting_review", mark_awaiting_review)
    builder.add_node("human_review", human_review)
    builder.add_node("revise_itinerary", revise_itinerary)
    builder.add_node("finalize_plan", finalize_plan)
    builder.add_node("mark_revision_limit", mark_revision_limit)

    # 2. Add edges for initial generation flow
    builder.add_edge(START, "validate_request")
    builder.add_conditional_edges(
        "validate_request",
        route_request_validation,
        {
            "research_agent": "research_agent",
            "end": END,
        },
    )
    builder.add_edge("research_agent", "planner_agent")
    builder.add_edge("planner_agent", "itinerary_validator")
    builder.add_edge("itinerary_validator", "mark_awaiting_review")
    builder.add_edge("mark_awaiting_review", "human_review")

    # 3. Add conditional edge for review outcome
    builder.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "finalize_plan": "finalize_plan",
            "revise_itinerary": "revise_itinerary",
            "mark_revision_limit": "mark_revision_limit",
        },
    )

    # 4. Revision loops back through validation to human review
    builder.add_edge("revise_itinerary", "itinerary_validator")

    # 5. Terminal edges
    builder.add_edge("finalize_plan", END)
    builder.add_edge("mark_revision_limit", END)

    return builder.compile(checkpointer=checkpointer)
