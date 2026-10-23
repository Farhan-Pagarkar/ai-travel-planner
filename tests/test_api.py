"""Tests for FastAPI endpoints exposing travel planning workflow."""

from pathlib import Path
from typing import Any
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client(tmp_path: Path):
    """Provide a TestClient using a temporary SQLite checkpoint database."""
    db_file = tmp_path / "test_api_checkpoints.sqlite"
    test_app = create_app(checkpoint_path=str(db_file))
    with TestClient(test_app) as test_client:
        yield test_client


@pytest.fixture
def sample_payload() -> dict[str, Any]:
    return {
        "destination": "Rome, Italy",
        "start_date": "2026-10-01",
        "end_date": "2026-10-04",
        "total_trip_budget": 1800.0,
        "currency": "EUR",
        "interests": ["architecture", "food", "history"],
        "traveler_count": 2,
        "preferences": {"pace": "relaxed"},
    }


def test_create_plan_returns_id_and_awaiting_review_draft(client: TestClient, sample_payload: dict[str, Any]):
    response = client.post("/plan", json=sample_payload)
    assert response.status_code == 200

    data = response.json()
    assert "plan_id" in data
    assert data["status"] == "awaiting_review"
    assert data["workflow_stage"] == "awaiting_review"
    assert data["validation_issues"] == []

    draft = data["draft_itinerary"]
    assert draft is not None
    assert draft["destination"] == "Rome, Italy"
    assert draft["total_days"] == 4
    assert len(draft["days"]) == 4


def test_get_plan_returns_saved_state(client: TestClient, sample_payload: dict[str, Any]):
    create_res = client.post("/plan", json=sample_payload)
    plan_id = create_res.json()["plan_id"]

    get_res = client.get(f"/plan/{plan_id}")
    assert get_res.status_code == 200

    data = get_res.json()
    assert data["plan_id"] == plan_id
    assert data["status"] == "awaiting_review"
    assert data["workflow_stage"] == "awaiting_review"
    assert data["revision_count"] == 0
    assert data["draft_itinerary"]["destination"] == "Rome, Italy"


def test_approve_plan_and_retrieve_final_itinerary(client: TestClient, sample_payload: dict[str, Any]):
    create_res = client.post("/plan", json=sample_payload)
    plan_id = create_res.json()["plan_id"]

    # Approve
    review_res = client.post(f"/plan/{plan_id}/review", json={"action": "approve"})
    assert review_res.status_code == 200

    review_data = review_res.json()
    assert review_data["status"] == "finalized"
    assert review_data["workflow_stage"] == "completed"
    assert review_data["final_itinerary"] is not None

    # Retrieve final itinerary
    final_res = client.get(f"/plan/{plan_id}/final")
    assert final_res.status_code == 200

    final_data = final_res.json()
    assert final_data["plan_id"] == plan_id
    assert final_data["status"] == "finalized"
    assert final_data["final_itinerary"]["destination"] == "Rome, Italy"


def test_reject_plan_produces_revised_draft_and_pauses(client: TestClient, sample_payload: dict[str, Any]):
    create_res = client.post("/plan", json=sample_payload)
    plan_id = create_res.json()["plan_id"]

    reject_payload = {
        "action": "reject",
        "feedback": "Include more time for walking tours in Trastevere.",
    }
    review_res = client.post(f"/plan/{plan_id}/review", json=reject_payload)
    assert review_res.status_code == 200

    data = review_res.json()
    assert data["status"] == "awaiting_review"
    assert data["workflow_stage"] == "awaiting_review"
    assert data["revision_count"] == 1

    revised_draft = data["draft_itinerary"]
    assert "Revision 1" in revised_draft["summary"]
    activities_text = " ".join(
        act for day in revised_draft["days"] for act in day["activities"]
    )
    assert "Trastevere" in activities_text


def test_modify_plan_incorporates_requested_change_and_pauses(
    client: TestClient, sample_payload: dict[str, Any]
):
    create_res = client.post("/plan", json=sample_payload)
    plan_id = create_res.json()["plan_id"]

    modify_payload = {
        "action": "modify",
        "modifications": "Add an evening pasta making class.",
    }
    review_res = client.post(f"/plan/{plan_id}/review", json=modify_payload)
    assert review_res.status_code == 200

    data = review_res.json()
    assert data["status"] == "awaiting_review"
    assert data["revision_count"] == 1

    activities_text = " ".join(
        act for day in data["draft_itinerary"]["days"] for act in day["activities"]
    )
    assert "pasta making class" in activities_text


def test_get_final_plan_before_approval_returns_409(client: TestClient, sample_payload: dict[str, Any]):
    create_res = client.post("/plan", json=sample_payload)
    plan_id = create_res.json()["plan_id"]

    final_res = client.get(f"/plan/{plan_id}/final")
    assert final_res.status_code == 409
    assert "not finalized" in final_res.json()["detail"].lower()


def test_unknown_plan_id_returns_404(client: TestClient):
    unknown_id = "00000000-0000-0000-0000-000000000000"

    get_res = client.get(f"/plan/{unknown_id}")
    assert get_res.status_code == 404

    review_res = client.post(f"/plan/{unknown_id}/review", json={"action": "approve"})
    assert review_res.status_code == 404

    final_res = client.get(f"/plan/{unknown_id}/final")
    assert final_res.status_code == 404


def test_invalid_review_payload_returns_422(client: TestClient, sample_payload: dict[str, Any]):
    create_res = client.post("/plan", json=sample_payload)
    plan_id = create_res.json()["plan_id"]

    # Reject without feedback
    res1 = client.post(f"/plan/{plan_id}/review", json={"action": "reject"})
    assert res1.status_code == 422

    # Reject with whitespace feedback
    res2 = client.post(f"/plan/{plan_id}/review", json={"action": "reject", "feedback": "   "})
    assert res2.status_code == 422

    # Modify without modifications
    res3 = client.post(f"/plan/{plan_id}/review", json={"action": "modify"})
    assert res3.status_code == 422

    # Unknown action
    res4 = client.post(f"/plan/{plan_id}/review", json={"action": "cancel"})
    assert res4.status_code == 422


def test_submitting_review_after_finalization_returns_409(
    client: TestClient, sample_payload: dict[str, Any]
):
    create_res = client.post("/plan", json=sample_payload)
    plan_id = create_res.json()["plan_id"]

    # First review: approve
    res_approve = client.post(f"/plan/{plan_id}/review", json={"action": "approve"})
    assert res_approve.status_code == 200

    # Second review: attempted after plan is already finalized
    res_second = client.post(f"/plan/{plan_id}/review", json={"action": "approve"})
    assert res_second.status_code == 409
    assert "not awaiting review" in res_second.json()["detail"].lower()


def test_create_plan_invalid_request_returns_422(client: TestClient):
    invalid_payload = {
        "destination": "Paris",
        "start_date": "2026-10-10",
        "end_date": "2026-10-01",  # Invalid date order
        "total_trip_budget": -100,  # Invalid budget
        "interests": [],
    }
    response = client.post("/plan", json=invalid_payload)
    assert response.status_code == 422
