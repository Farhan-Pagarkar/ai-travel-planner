from datetime import date, timedelta

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
        "start_date": "2026-11-01",
        "end_date": "2026-11-04",
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
        act.get("description", "") if isinstance(act, dict) else str(act)
        for day in revised_draft["days"]
        for act in day["activities"]
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
        act.get("description", "") if isinstance(act, dict) else str(act)
        for day in data["draft_itinerary"]["days"]
        for act in day["activities"]
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


def test_api_enforces_revision_limit(client: TestClient, sample_payload: dict[str, Any]):
    """After 3 revisions without approval, the workflow terminates at revision limit and cannot be finalized."""
    create_res = client.post("/plan", json=sample_payload)
    plan_id = create_res.json()["plan_id"]
    assert create_res.json()["status"] == "awaiting_review"

    # Revision 1
    rev1 = client.post(f"/plan/{plan_id}/review", json={"action": "reject", "feedback": "Need more food"})
    assert rev1.status_code == 200
    assert rev1.json()["status"] == "awaiting_review"
    assert rev1.json()["revision_count"] == 1

    # Revision 2
    rev2 = client.post(f"/plan/{plan_id}/review", json={"action": "reject", "feedback": "Need more walks"})
    assert rev2.status_code == 200
    assert rev2.json()["status"] == "awaiting_review"
    assert rev2.json()["revision_count"] == 2

    # Revision 3
    rev3 = client.post(f"/plan/{plan_id}/review", json={"action": "reject", "feedback": "Different order"})
    assert rev3.status_code == 200
    assert rev3.json()["status"] == "awaiting_review"
    assert rev3.json()["revision_count"] == 3

    # Revision 4: Reaches limit (MAX_REVISIONS = 3)
    rev4 = client.post(f"/plan/{plan_id}/review", json={"action": "reject", "feedback": "Another change"})
    assert rev4.status_code == 200
    data4 = rev4.json()
    assert data4["status"] == "revision_limit_reached"
    assert data4["workflow_stage"] == "ended_revision_limit"

    # Cannot retrieve final plan
    res_final = client.get(f"/plan/{plan_id}/final")
    assert res_final.status_code == 409
    assert "not finalized" in res_final.json()["detail"].lower()

    # Subsequent review attempt fails with 409
    res_after = client.post(f"/plan/{plan_id}/review", json={"action": "approve"})
    assert res_after.status_code == 409
    assert "not awaiting review" in res_after.json()["detail"].lower()


def test_frontend_routes_served(client: TestClient):
    """Verify that the frontend HTML, CSS, and JS are properly served by FastAPI."""
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "text/html" in res_index.headers["content-type"]
    assert "AI Travel Planner" in res_index.text
    assert "plan-form" in res_index.text

    res_html = client.get("/index.html")
    assert res_html.status_code == 200
    assert "text/html" in res_html.headers["content-type"]

    res_css = client.get("/static/styles.css")
    assert res_css.status_code == 200
    assert "app-header" in res_css.text

    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
    assert "currentPlanId" in res_js.text


def test_create_plan_returns_controlled_502_on_llm_failure(
    client: TestClient, sample_payload: dict[str, Any], monkeypatch
):
    """When itinerary generation fails permanently, the API returns HTTP 502 with sanitized detail instead of HTTP 500."""
    from app.agents.planner import PlannerError

    def failing_planner(*args, **kwargs):
        raise PlannerError("Groq itinerary generation failed after retry: Error code: 400 - Failed to validate JSON. Please adjust your prompt. code: json_validate_failed")

    monkeypatch.setattr("app.graph.nodes.run_planner_agent", failing_planner)
    monkeypatch.setattr("app.agents.planner.run_planner_agent", failing_planner)

    res = client.post("/plan", json=sample_payload)
    assert res.status_code == 502
    assert "Unable to generate travel plan" in res.json()["detail"]
    assert "json_validate_failed" in res.json()["detail"]


def test_create_plan_sanitizes_secrets_in_error_response(
    client: TestClient, sample_payload: dict[str, Any], monkeypatch
):
    """When an error contains secret API keys or headers, they are scrubbed from the response."""
    from app.agents.planner import PlannerError

    secret_key = "gsk_live_api_secret_key_123456789"
    def failing_planner(*args, **kwargs):
        raise PlannerError(f"Groq API call failed with key {secret_key}")

    monkeypatch.setattr("app.graph.nodes.run_planner_agent", failing_planner)
    monkeypatch.setattr("app.agents.planner.run_planner_agent", failing_planner)

    res = client.post("/plan", json=sample_payload)
    assert res.status_code == 502
    assert secret_key not in res.json()["detail"]
    assert "[REDACTED_API_KEY]" in res.json()["detail"]


def test_favicon_route(client: TestClient):
    """Verify favicon endpoint returns 204."""
    res = client.get("/favicon.ico")
    assert res.status_code == 204


def test_create_plan_rejects_past_start_date(client: TestClient, sample_payload: dict[str, Any]):
    """Requests with start_date earlier than today are rejected with 422 before external calls."""
    payload = dict(sample_payload)
    payload["start_date"] = (date.today() - timedelta(days=1)).isoformat()
    payload["end_date"] = (date.today() + timedelta(days=3)).isoformat()

    res = client.post("/plan", json=payload)
    assert res.status_code == 422
    err_msg = str(res.json())
    assert "cannot be earlier than the current date" in err_msg or "cannot be in the past" in err_msg


def test_create_plan_accepts_today_start_date(client: TestClient, sample_payload: dict[str, Any]):
    """Requests with start_date equal to server's today date are accepted."""
    payload = dict(sample_payload)
    payload["start_date"] = date.today().isoformat()
    payload["end_date"] = (date.today() + timedelta(days=3)).isoformat()

    res = client.post("/plan", json=payload)
    assert res.status_code == 200
    assert res.json()["status"] == "awaiting_review"


def test_create_plan_accepts_valid_14_day_trip(client: TestClient, sample_payload: dict[str, Any]):
    """A trip of exactly 14 calendar days inclusive is accepted."""
    payload = dict(sample_payload)
    start = date.today() + timedelta(days=1)
    payload["start_date"] = start.isoformat()
    payload["end_date"] = (start + timedelta(days=13)).isoformat()  # 14 days inclusive

    res = client.post("/plan", json=payload)
    assert res.status_code == 200
    assert res.json()["status"] == "awaiting_review"


def test_create_plan_rejects_trip_exceeding_14_days(client: TestClient, sample_payload: dict[str, Any]):
    """A trip exceeding 14 calendar days inclusive is rejected with 422."""
    payload = dict(sample_payload)
    start = date.today() + timedelta(days=1)
    payload["start_date"] = start.isoformat()
    payload["end_date"] = (start + timedelta(days=14)).isoformat()  # 15 days inclusive

    res = client.post("/plan", json=payload)
    assert res.status_code == 422
    err_msg = str(res.json())
    assert "exceeds the maximum allowed limit of 14 calendar days" in err_msg
