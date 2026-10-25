from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request, status
from app.schemas.request import TravelRequest
from app.schemas.review import HumanReviewInput
from app.services import plan_service

router = APIRouter(prefix="/plan", tags=["plans"])
def get_graph(request: Request) -> Any:
    #Dependency retrieving the compiled LangGraph workflow from application state.
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Workflow graph is not initialized.",
        )
    return graph


@router.post("", status_code=status.HTTP_200_OK)
@router.post("/", status_code=status.HTTP_200_OK, include_in_schema=False)
def create_plan(request: TravelRequest, graph: Any = Depends(get_graph)) -> dict[str, Any]:
    #Create a new travel plan and run until human review.
    return plan_service.create_plan(graph=graph, request=request)


@router.get("/{plan_id}", status_code=status.HTTP_200_OK)
def get_plan(plan_id: str, graph: Any = Depends(get_graph)) -> dict[str, Any]:
    #Retrieve saved state of an existing plan.
    return plan_service.get_plan(graph=graph, plan_id=plan_id)


@router.post("/{plan_id}/review", status_code=status.HTTP_200_OK)
def review_plan(
    plan_id: str,
    review: HumanReviewInput,
    graph: Any = Depends(get_graph),
) -> dict[str, Any]:
    #Submit human review decision to resume plan execution.
    return plan_service.review_plan(graph=graph, plan_id=plan_id, review=review)


@router.get("/{plan_id}/final", status_code=status.HTTP_200_OK)
def get_final_plan(plan_id: str, graph: Any = Depends(get_graph)) -> dict[str, Any]:
    #Retrieve finalized itinerary for an approved plan.
    return plan_service.get_final_plan(graph=graph, plan_id=plan_id)
