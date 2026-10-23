"""Tests for LangGraph travel planner workflow, checkpointing, and HITL interrupts."""

from pathlib import Path
from typing import Any
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import ValidationError

from app.graph.workflow import MAX_REVISIONS, create_travel_planner_graph
from app.schemas.review import HumanReviewInput


@pytest.fixture
def sample_travel_request() -> dict[str, Any]:
    return {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-04",
        "total_trip_budget": 2500.0,
        "currency": "USD",
        "interests": ["temples", "culinary", "gardens"],
        "traveler_count": 2,
        "preferences": {"pace": "moderate"},
    }


def test_plan_start_reaches_human_review_interrupt(tmp_path: Path, sample_travel_request: dict[str, Any]):
    db_file = tmp_path / "test_checkpoints.sqlite"
    plan_id = "plan-start-test-01"

    with SqliteSaver.from_conn_string(str(db_file)) as saver:
        graph = create_travel_planner_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": plan_id}}

        result = graph.invoke({"request": sample_travel_request}, config=config)

        # Execution should be paused with an interrupt
        assert "__interrupt__" in result
        state = graph.get_state(config)

        assert state.values["status"] == "awaiting_review"
        assert state.values["workflow_stage"] == "awaiting_review"
        assert state.values["revision_count"] == 0
        assert state.values["validation_issues"] == []

        draft = state.values["draft_itinerary"]
        assert draft is not None
        assert draft["destination"] == "Kyoto, Japan"
        assert draft["total_days"] == 4
        assert len(draft["days"]) == 4
        assert draft["total_estimated_cost"] <= sample_travel_request["total_trip_budget"]
        assert state.values.get("final_itinerary") is None


def test_approve_draft_resumes_and_finalizes(tmp_path: Path, sample_travel_request: dict[str, Any]):
    db_file = tmp_path / "test_checkpoints.sqlite"
    plan_id = "plan-approve-test-02"

    with SqliteSaver.from_conn_string(str(db_file)) as saver:
        graph = create_travel_planner_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": plan_id}}

        # 1. Start and pause at review
        graph.invoke({"request": sample_travel_request}, config=config)

        # 2. Resume with approval
        approval_command = Command(resume={"action": "approve"})
        final_result = graph.invoke(approval_command, config=config)

        assert final_result["status"] == "finalized"
        assert final_result["workflow_stage"] == "completed"
        assert final_result["review_action"] == "approve"
        assert final_result["final_itinerary"] is not None
        assert final_result["final_itinerary"] == final_result["draft_itinerary"]

        # Thread execution has completed
        state = graph.get_state(config)
        assert state.next == ()


def test_reject_with_feedback_updates_draft_and_returns_to_review(
    tmp_path: Path, sample_travel_request: dict[str, Any]
):
    db_file = tmp_path / "test_checkpoints.sqlite"
    plan_id = "plan-reject-test-03"

    with SqliteSaver.from_conn_string(str(db_file)) as saver:
        graph = create_travel_planner_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": plan_id}}

        # 1. Initial generation
        graph.invoke({"request": sample_travel_request}, config=config)

        # 2. Reject with specific feedback
        reject_payload = {
            "action": "reject",
            "feedback": "Budget is fine but include more morning temple walks.",
        }
        res = graph.invoke(Command(resume=reject_payload), config=config)

        # Pauses again for human review
        assert "__interrupt__" in res
        state = graph.get_state(config)

        assert state.values["status"] == "awaiting_review"
        assert state.values["workflow_stage"] == "awaiting_review"
        assert state.values["revision_count"] == 1

        revised_draft = state.values["draft_itinerary"]
        assert revised_draft is not None
        assert "Revision 1" in revised_draft["summary"]
        # Check that feedback is recorded in the activity list
        activities_text = " ".join(
            act for day in revised_draft["days"] for act in day["activities"]
        )
        assert "Budget is fine but include more morning temple walks" in activities_text


def test_revised_draft_can_subsequently_be_approved(tmp_path: Path, sample_travel_request: dict[str, Any]):
    db_file = tmp_path / "test_checkpoints.sqlite"
    plan_id = "plan-revise-approve-test-04"

    with SqliteSaver.from_conn_string(str(db_file)) as saver:
        graph = create_travel_planner_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": plan_id}}

        # 1. Initial draft
        graph.invoke({"request": sample_travel_request}, config=config)

        # 2. Reject once
        graph.invoke(Command(resume={"action": "reject", "feedback": "Need slower pace"}), config=config)

        # 3. Approve the revised draft
        res = graph.invoke(Command(resume={"action": "approve"}), config=config)

        assert res["status"] == "finalized"
        assert res["workflow_stage"] == "completed"
        assert res["final_itinerary"] is not None
        assert "Revision 1" in res["final_itinerary"]["summary"]


def test_modifications_incorporated_into_revised_draft(
    tmp_path: Path, sample_travel_request: dict[str, Any]
):
    db_file = tmp_path / "test_checkpoints.sqlite"
    plan_id = "plan-modify-test-05"

    with SqliteSaver.from_conn_string(str(db_file)) as saver:
        graph = create_travel_planner_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": plan_id}}

        graph.invoke({"request": sample_travel_request}, config=config)

        mod_payload = {
            "action": "modify",
            "modifications": "Swap dinner for a private culinary cooking class on Day 2.",
        }
        res = graph.invoke(Command(resume=mod_payload), config=config)

        assert "__interrupt__" in res
        state = graph.get_state(config)
        assert state.values["revision_count"] == 1
        assert state.values["requested_modifications"] == mod_payload["modifications"]

        activities_text = " ".join(
            act for day in state.values["draft_itinerary"]["days"] for act in day["activities"]
        )
        assert "private culinary cooking class" in activities_text


def test_invalid_review_payloads_rejected(tmp_path: Path, sample_travel_request: dict[str, Any]):
    # Schema-level rejections
    with pytest.raises(ValidationError):
        HumanReviewInput.model_validate({"action": "reject"})  # Missing feedback

    with pytest.raises(ValidationError):
        HumanReviewInput.model_validate({"action": "reject", "feedback": "   "})  # Blank feedback

    with pytest.raises(ValidationError):
        HumanReviewInput.model_validate({"action": "modify"})  # Missing modifications

    with pytest.raises(ValidationError):
        HumanReviewInput.model_validate({"action": "invalid_action"})

    # Workflow-level rejection when resuming
    db_file = tmp_path / "test_checkpoints.sqlite"
    plan_id = "plan-invalid-review-test-06"

    with SqliteSaver.from_conn_string(str(db_file)) as saver:
        graph = create_travel_planner_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": plan_id}}

        graph.invoke({"request": sample_travel_request}, config=config)

        with pytest.raises(ValidationError):
            graph.invoke(Command(resume={"action": "reject"}), config=config)


def test_checkpoint_resumed_after_closing_and_reopening_sqlite(
    tmp_path: Path, sample_travel_request: dict[str, Any]
):
    db_file = tmp_path / "persisted_checkpoints.sqlite"
    plan_id = "plan-reopen-test-07"
    config = {"configurable": {"thread_id": plan_id}}

    # Session 1: Run up to human-review interrupt, then close SQLite connection
    with SqliteSaver.from_conn_string(str(db_file)) as saver1:
        graph1 = create_travel_planner_graph(checkpointer=saver1)
        graph1.invoke({"request": sample_travel_request}, config=config)

    # Session 2: Open new connection to the same SQLite file and resume execution
    with SqliteSaver.from_conn_string(str(db_file)) as saver2:
        graph2 = create_travel_planner_graph(checkpointer=saver2)

        # Verify state is recovered cleanly
        reloaded_state = graph2.get_state(config)
        assert reloaded_state.values["status"] == "awaiting_review"
        assert reloaded_state.values["draft_itinerary"]["destination"] == "Kyoto, Japan"

        # Resume from checkpoint
        result = graph2.invoke(Command(resume={"action": "approve"}), config=config)

        assert result["status"] == "finalized"
        assert result["final_itinerary"] is not None
        assert result["final_itinerary"]["destination"] == "Kyoto, Japan"


def test_workflow_cannot_finalize_without_approval(
    tmp_path: Path, sample_travel_request: dict[str, Any]
):
    db_file = tmp_path / "test_checkpoints.sqlite"
    plan_id = "plan-revision-limit-test-08"

    with SqliteSaver.from_conn_string(str(db_file)) as saver:
        graph = create_travel_planner_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": plan_id}}

        # Start (revision 0)
        graph.invoke({"request": sample_travel_request}, config=config)
        state_init = graph.get_state(config)
        assert state_init.values.get("final_itinerary") is None

        # Rejection 1 -> produces Revision 1
        graph.invoke(Command(resume={"action": "reject", "feedback": "Change 1"}), config=config)
        assert graph.get_state(config).values["revision_count"] == 1
        assert graph.get_state(config).values.get("final_itinerary") is None

        # Rejection 2 -> produces Revision 2
        graph.invoke(Command(resume={"action": "reject", "feedback": "Change 2"}), config=config)
        assert graph.get_state(config).values["revision_count"] == 2
        assert graph.get_state(config).values.get("final_itinerary") is None

        # Rejection 3 -> produces Revision 3
        graph.invoke(Command(resume={"action": "reject", "feedback": "Change 3"}), config=config)
        assert graph.get_state(config).values["revision_count"] == 3
        assert graph.get_state(config).values.get("final_itinerary") is None

        # Rejection 4: MAX_REVISIONS (3) is reached, should terminate without finalization
        final_result = graph.invoke(
            Command(resume={"action": "reject", "feedback": "Change 4"}), config=config
        )

        assert final_result["status"] == "revision_limit_reached"
        assert final_result["workflow_stage"] == "ended_revision_limit"
        assert final_result.get("final_itinerary") is None

        # Confirms workflow has ended
        assert graph.get_state(config).next == ()


def test_invalid_travel_request_stops_at_validation(tmp_path: Path):
    db_file = tmp_path / "test_checkpoints.sqlite"
    plan_id = "plan-invalid-request-test-09"

    invalid_request = {
        "destination": "Kyoto",
        "start_date": "2026-11-10",
        "end_date": "2026-11-01",  # Invalid: end before start
        "total_trip_budget": -500.0,  # Invalid: negative budget
        "currency": "USD",
        "interests": [],  # Invalid: empty interests
    }

    with SqliteSaver.from_conn_string(str(db_file)) as saver:
        graph = create_travel_planner_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": plan_id}}

        res = graph.invoke({"request": invalid_request}, config=config)

        assert res["status"] == "error"
        assert res["workflow_stage"] == "validation"
        assert len(res["errors"]) > 0
        assert res.get("draft_itinerary") is None
        assert graph.get_state(config).next == ()
