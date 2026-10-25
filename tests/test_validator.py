"""Tests for the Itinerary Validator Agent.

Tests cover:
- Deterministic checks: dates, duration, numbering, cost math, budget ceilings,
  candidate grounding, source URLs, allowed generic activities, and duplicates.
- Safe monetary parsing (negative, non-finite, string, or missing values).
- Groq-powered advisory review for safety, weather, and schedule feasibility.
- Resilient failure handling: preserving deterministic findings on Groq errors.
- Integration into LangGraph human-review interrupt payload.
"""

import json
from unittest.mock import MagicMock
import pytest

from app.agents.itinerary_validator import (
    _run_deterministic_checks,
    _run_groq_quality_review,
    run_itinerary_validator,
)
from app.graph.workflow import create_travel_planner_graph
from langgraph.checkpoint.memory import MemorySaver


@pytest.fixture
def sample_request() -> dict:
    return {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-03",
        "total_trip_budget": 1500.0,
        "currency": "USD",
        "interests": ["temples", "culinary"],
        "traveler_count": 2,
        "preferences": {"pace": "moderate"},
    }


@pytest.fixture
def sample_research() -> dict:
    return {
        "destination": "Kyoto, Japan",
        "attractions": [
            {
                "name": "Kiyomizu-dera Temple",
                "description": "Historic Buddhist temple overlooking eastern Kyoto.",
                "source_url": "https://example.com/kiyomizu",
            },
            {
                "name": "Fushimi Inari Shrine",
                "description": "Shinto shrine with thousands of red torii gates.",
                "source_url": "https://example.com/fushimi",
            },
        ],
        "safety_considerations": ["Avoid crowded buses during peak morning hours."],
        "seasonal_considerations": ["Peak autumn foliage with high tourist traffic."],
        "recent_local_news": [
            {
                "title": "East Kyoto Bus Lane Closure",
                "summary": "Main bus lane near Kiyomizu-dera undergoing repairs on Nov 2.",
                "publication_date": "2026-10-25",
                "potential_travel_impact": "high",
            }
        ],
        "upcoming_local_events": [
            {
                "title": "Evening Garden Illumination",
                "summary": "Special night light-up at Kodai-ji Temple.",
                "source_url": "https://example.com/events/lightup",
                "event_date": "2026-11-01 to 2026-11-05",
            }
        ],
        "weather": {
            "forecast_days": [
                {
                    "date": "2026-11-01",
                    "temperature_max_c": 18.0,
                    "temperature_min_c": 10.0,
                    "precipitation_probability_pct": 10,
                    "weather_condition": "Clear",
                },
                {
                    "date": "2026-11-02",
                    "temperature_max_c": 14.0,
                    "temperature_min_c": 8.0,
                    "precipitation_probability_pct": 85,
                    "weather_condition": "Heavy Rain",
                },
                {
                    "date": "2026-11-03",
                    "temperature_max_c": 16.0,
                    "temperature_min_c": 9.0,
                    "precipitation_probability_pct": 15,
                    "weather_condition": "Partly Cloudy",
                },
            ],
            "notes": "Heavy rain expected on Day 2.",
        },
    }


@pytest.fixture
def valid_itinerary() -> dict:
    return {
        "destination": "Kyoto, Japan",
        "total_days": 3,
        "currency": "USD",
        "total_estimated_cost": 300.0,
        "days": [
            {
                "day": 1,
                "date": "2026-11-01",
                "focus": "temples",
                "activities": [
                    {
                        "name": "Kiyomizu-dera Temple",
                        "description": "Historic wooden temple terrace.",
                        "category": "attraction",
                        "start_time": "09:00 AM",
                        "estimated_duration": "2.5 hours",
                        "estimated_cost": 50.0,
                        "candidate_id": "attr-1",
                        "source_url": "https://example.com/kiyomizu",
                    },
                    {
                        "name": "Traditional Japanese Lunch",
                        "description": "Lunch at local restaurant.",
                        "category": "meal",
                        "start_time": "12:30 PM",
                        "estimated_duration": "1.5 hours",
                        "estimated_cost": 50.0,
                        "candidate_id": None,
                        "source_url": None,
                    },
                ],
                "estimated_cost": 100.0,
            },
            {
                "day": 2,
                "date": "2026-11-02",
                "focus": "shrine and tea",
                "activities": [
                    {
                        "name": "Fushimi Inari Shrine",
                        "description": "Hike the scenic torii path.",
                        "category": "attraction",
                        "start_time": "09:00 AM",
                        "estimated_duration": "3 hours",
                        "estimated_cost": 50.0,
                        "candidate_id": "attr-2",
                        "source_url": "https://example.com/fushimi",
                    },
                    {
                        "name": "Tea House Relaxation",
                        "description": "Matcha tea in Gion.",
                        "category": "leisure",
                        "start_time": "03:00 PM",
                        "estimated_duration": "1.5 hours",
                        "estimated_cost": 50.0,
                        "candidate_id": None,
                        "source_url": None,
                    },
                ],
                "estimated_cost": 100.0,
            },
            {
                "day": 3,
                "date": "2026-11-03",
                "focus": "evening event",
                "activities": [
                    {
                        "name": "Evening Garden Illumination",
                        "description": "Temple night illumination.",
                        "category": "event",
                        "start_time": "06:00 PM",
                        "estimated_duration": "2 hours",
                        "estimated_cost": 50.0,
                        "candidate_id": "event-1",
                        "source_url": "https://example.com/events/lightup",
                    },
                    {
                        "name": "Farewell Dinner",
                        "description": "Kaiseki dining.",
                        "category": "dining",
                        "start_time": "08:30 PM",
                        "estimated_duration": "2 hours",
                        "estimated_cost": 50.0,
                        "candidate_id": None,
                        "source_url": None,
                    },
                ],
                "estimated_cost": 100.0,
            },
        ],
        "summary": "3-day Kyoto itinerary.",
        "assumptions": ["Moderate pace."],
        "warnings": [],
    }


def test_valid_itinerary_no_deterministic_issues(sample_request, sample_research, valid_itinerary):
    """A valid itinerary passes all deterministic checks with zero issues."""
    issues = _run_deterministic_checks(
        request=sample_request,
        draft_itinerary=valid_itinerary,
        research=sample_research,
    )
    assert issues == []

    # Full validator with mock Groq returning empty list also returns empty list
    mock_groq = MagicMock()
    choice = MagicMock()
    choice.message.content = json.dumps({"issues": []})
    mock_groq.chat.completions.create.return_value = MagicMock(choices=[choice])

    all_issues = run_itinerary_validator(
        request=sample_request,
        draft_itinerary=valid_itinerary,
        research=sample_research,
        groq_client=mock_groq,
    )
    assert all_issues == []


def test_missing_duplicate_and_out_of_order_dates(sample_request, sample_research, valid_itinerary):
    """Missing, duplicate, or out-of-order dates are detected deterministically."""
    # 1. Missing day (only 2 days instead of 3)
    bad_itin_missing = dict(valid_itinerary)
    bad_itin_missing["days"] = valid_itinerary["days"][:2]
    bad_itin_missing["total_days"] = 2
    bad_itin_missing["total_estimated_cost"] = 200.0
    issues = _run_deterministic_checks(sample_request, bad_itin_missing, sample_research)
    assert any("duration mismatch" in i.lower() for i in issues)
    assert any("day count mismatch" in i.lower() for i in issues)

    # 2. Duplicate day dates
    bad_itin_dup = json.loads(json.dumps(valid_itinerary))
    bad_itin_dup["days"][1]["date"] = "2026-11-01"  # Duplicate date
    issues_dup = _run_deterministic_checks(sample_request, bad_itin_dup, sample_research)
    assert any("duplicate day dates" in i.lower() for i in issues_dup)

    # 3. Out-of-order dates
    bad_itin_order = json.loads(json.dumps(valid_itinerary))
    bad_itin_order["days"][0]["date"] = "2026-11-02"
    bad_itin_order["days"][1]["date"] = "2026-11-01"
    issues_order = _run_deterministic_checks(sample_request, bad_itin_order, sample_research)
    assert any("date coverage mismatch" in i.lower() for i in issues_order)


def test_incorrect_day_numbering_or_duration(sample_request, sample_research, valid_itinerary):
    """Non-sequential day numbers and duration mismatches are flagged."""
    bad_itin = json.loads(json.dumps(valid_itinerary))
    bad_itin["days"][1]["day"] = 4  # Non-sequential: 1, 4, 3
    issues = _run_deterministic_checks(sample_request, bad_itin, sample_research)
    assert any("day numbering error" in i.lower() for i in issues)


def test_cost_arithmetic_mismatches(sample_request, sample_research, valid_itinerary):
    """Activity-to-day and day-to-total cost discrepancies are caught."""
    # Activity cost mismatch on Day 1: 50 + 50 = 100, but day estimated_cost is 150
    bad_day_cost = json.loads(json.dumps(valid_itinerary))
    bad_day_cost["days"][0]["estimated_cost"] = 150.0
    bad_day_cost["total_estimated_cost"] = 350.0
    issues = _run_deterministic_checks(sample_request, bad_day_cost, sample_research)
    assert any("day 1 cost mismatch" in i.lower() for i in issues)

    # Day-to-total mismatch: days sum to 300, but total_estimated_cost reported is 400
    bad_total_cost = json.loads(json.dumps(valid_itinerary))
    bad_total_cost["total_estimated_cost"] = 400.0
    issues_total = _run_deterministic_checks(sample_request, bad_total_cost, sample_research)
    assert any("total cost mismatch" in i.lower() for i in issues_total)


def test_negative_non_finite_or_malformed_costs(sample_request, sample_research, valid_itinerary):
    """Negative, non-finite, missing, or malformed costs are rejected safely."""
    bad_cost_itin = json.loads(json.dumps(valid_itinerary))
    bad_cost_itin["days"][0]["activities"][0]["estimated_cost"] = -25.0
    bad_cost_itin["days"][1]["activities"][0]["estimated_cost"] = "not_a_number"
    bad_cost_itin["days"][2]["activities"][0]["estimated_cost"] = float("nan")

    issues = _run_deterministic_checks(sample_request, bad_cost_itin, sample_research)
    assert any("negative" in i.lower() or "invalid" in i.lower() for i in issues)


def test_budget_overruns(sample_request, sample_research, valid_itinerary):
    """Total estimated cost exceeding total trip budget is flagged as an error."""
    expensive_itin = json.loads(json.dumps(valid_itinerary))
    expensive_itin["total_estimated_cost"] = 2500.0  # Budget is 1500.0
    expensive_itin["days"][0]["estimated_cost"] = 1000.0
    expensive_itin["days"][0]["activities"][0]["estimated_cost"] = 950.0
    expensive_itin["days"][0]["activities"][1]["estimated_cost"] = 50.0
    expensive_itin["days"][1]["estimated_cost"] = 1000.0
    expensive_itin["days"][1]["activities"][0]["estimated_cost"] = 950.0
    expensive_itin["days"][1]["activities"][1]["estimated_cost"] = 50.0
    expensive_itin["days"][2]["estimated_cost"] = 500.0
    expensive_itin["days"][2]["activities"][0]["estimated_cost"] = 450.0
    expensive_itin["days"][2]["activities"][1]["estimated_cost"] = 50.0

    issues = _run_deterministic_checks(sample_request, expensive_itin, sample_research)
    assert any("budget exceeded" in i.lower() for i in issues)


def test_unknown_candidate_ids_and_incorrect_source_urls(sample_request, sample_research, valid_itinerary):
    """Unknown candidate IDs, modified source URLs, or URLs on generic activities are caught."""
    # 1. Unknown candidate ID
    bad_cand = json.loads(json.dumps(valid_itinerary))
    bad_cand["days"][0]["activities"][0]["candidate_id"] = "attr-999"
    issues = _run_deterministic_checks(sample_request, bad_cand, sample_research)
    assert any("unknown candidate id 'attr-999'" in i.lower() for i in issues)

    # 2. Fabricated source URL for legitimate candidate
    bad_url = json.loads(json.dumps(valid_itinerary))
    bad_url["days"][0]["activities"][0]["source_url"] = "https://fabricated-domain.com/fake"
    issues_url = _run_deterministic_checks(sample_request, bad_url, sample_research)
    assert any("does not match verified candidate" in i.lower() for i in issues_url)

    # 3. Generic activity with source URL
    bad_generic = json.loads(json.dumps(valid_itinerary))
    bad_generic["days"][0]["activities"][1]["source_url"] = "https://example.com/cafe"
    issues_gen = _run_deterministic_checks(sample_request, bad_generic, sample_research)
    assert any("without a verified candidate id" in i.lower() for i in issues_gen)


def test_ungrounded_attraction_event_categories(sample_request, sample_research, valid_itinerary):
    """Activities categorized as attractions/events without candidate IDs are rejected."""
    bad_cat = json.loads(json.dumps(valid_itinerary))
    bad_cat["days"][0]["activities"][0]["candidate_id"] = None
    bad_cat["days"][0]["activities"][0]["source_url"] = None

    issues = _run_deterministic_checks(sample_request, bad_cat, sample_research)
    assert any("classified as an attraction/event" in i.lower() for i in issues)


def test_allowed_generic_meal_and_leisure_activities(sample_request, sample_research, valid_itinerary):
    """Generic meals, leisure, and transit blocks without candidate IDs are fully permitted."""
    issues = _run_deterministic_checks(sample_request, valid_itinerary, sample_research)
    # The valid itinerary includes generic "meal", "leisure", and "dining" activities with candidate_id=None
    assert issues == []


def test_duplicate_activities_within_a_day(sample_request, sample_research, valid_itinerary):
    """Duplicate activity scheduled twice in the same day triggers a warning."""
    dup_itin = json.loads(json.dumps(valid_itinerary))
    # Add identical attraction activity twice to Day 1
    dup_act = dict(dup_itin["days"][0]["activities"][0])
    dup_itin["days"][0]["activities"].append(dup_act)
    dup_itin["days"][0]["estimated_cost"] = 150.0
    dup_itin["total_estimated_cost"] = 350.0

    issues = _run_deterministic_checks(sample_request, dup_itin, sample_research)
    assert any("duplicate scheduled activity" in i.lower() for i in issues)
    assert any("[warning]" in i.lower() for i in issues)


def test_groq_quality_review_research_aware_checks(sample_request, sample_research, valid_itinerary):
    """Groq review detects safety, weather, and scheduling concerns from research."""
    mock_groq = MagicMock()
    choice = MagicMock()
    choice.message.content = json.dumps({
        "issues": [
            {
                "severity": "WARNING",
                "message": "Day 2 outdoor hiking at Fushimi Inari conflicts with 85% forecast precipitation chance.",
            },
            {
                "severity": "INFO",
                "message": "Day 1 bus travel may be delayed due to East Kyoto bus lane repairs reported in recent news.",
            },
        ]
    })
    mock_groq.chat.completions.create.return_value = MagicMock(choices=[choice])

    all_issues = run_itinerary_validator(
        request=sample_request,
        draft_itinerary=valid_itinerary,
        research=sample_research,
        groq_client=mock_groq,
    )

    assert len(all_issues) == 2
    assert any("[WARNING] Day 2 outdoor hiking" in i for i in all_issues)
    assert any("[INFO] Day 1 bus travel" in i for i in all_issues)

    # Verify that Groq was called with formatted evidence
    call_args = mock_groq.chat.completions.create.call_args[1]
    user_prompt = call_args["messages"][1]["content"]
    assert "Fushimi Inari Shrine" in user_prompt
    assert "East Kyoto Bus Lane Closure" in user_prompt
    assert "85%" in user_prompt


def test_invalid_json_or_failed_groq_preserves_deterministic_findings(
    sample_request, sample_research, valid_itinerary
):
    """If Groq quality review fails or outputs malformed JSON, deterministic findings are preserved with a clear warning."""
    # Introduce a deterministic error: budget exceeded
    broken_itin = json.loads(json.dumps(valid_itinerary))
    broken_itin["total_estimated_cost"] = 2000.0  # > 1500.0

    mock_groq = MagicMock()
    mock_groq.chat.completions.create.side_effect = RuntimeError("Groq rate limit exceeded")

    issues = run_itinerary_validator(
        request=sample_request,
        draft_itinerary=broken_itin,
        research=sample_research,
        groq_client=mock_groq,
    )

    # Deterministic budget issue must be preserved
    assert any("budget exceeded" in i.lower() for i in issues)
    # Clear warning that LLM review could not be completed
    assert any("llm quality review could not be completed" in i.lower() for i in issues)


def test_validator_integration_with_workflow_human_review_payload(sample_request):
    """Validation issues appear in the human-review interrupt payload."""
    checkpointer = MemorySaver()
    graph = create_travel_planner_graph(checkpointer=checkpointer)

    thread_id = "test-thread-validator-1"
    config = {"configurable": {"thread_id": thread_id}}

    # Run workflow until human review interrupt
    result = graph.invoke({"request": sample_request}, config=config)

    # In LangGraph interrupt state, __interrupt__ contains the interrupt payload
    interrupts = result.get("__interrupt__", ())
    assert len(interrupts) > 0
    payload = interrupts[0].value

    assert "validation_issues" in payload
    assert isinstance(payload["validation_issues"], list)
    # The valid mock draft produces an empty or clean list of issues
    assert payload["draft_itinerary"] is not None
    assert payload["revision_count"] == 0


def test_validator_detects_mismatched_activity_name_with_valid_id_and_url(
    sample_request, sample_research, valid_itinerary
):
    """Validator Agent catches when a valid candidate ID and URL are used for an unrelated activity name."""
    itin = json.loads(json.dumps(valid_itinerary))
    # attr-1 in sample_research is 'Kiyomizu-dera Temple', with url 'https://example.com/kiyomizu'
    itin["days"][0]["activities"][0]["name"] = "Completely Fabricated Theme Park"
    itin["days"][0]["activities"][0]["candidate_id"] = "attr-1"
    itin["days"][0]["activities"][0]["source_url"] = "https://example.com/kiyomizu"

    issues = _run_deterministic_checks(
        request=sample_request,
        draft_itinerary=itin,
        research=sample_research,
    )

    assert any(
        "Day 1 activity 'Completely Fabricated Theme Park' does not match verified candidate 'attr-1' name 'Kiyomizu-dera Temple'" in issue
        for issue in issues
    )


def test_validator_detects_missing_destination_and_currency(
    sample_request, sample_research, valid_itinerary
):
    """Missing destination or currency in request or draft itinerary must be flagged rather than treated as valid."""
    # 1. Missing request destination
    req_no_dest = dict(sample_request)
    req_no_dest["destination"] = ""
    issues = _run_deterministic_checks(
        request=req_no_dest,
        draft_itinerary=valid_itinerary,
        research=sample_research,
    )
    assert any("Travel request destination is missing" in i for i in issues)

    # 2. Missing itinerary destination
    itin_no_dest = json.loads(json.dumps(valid_itinerary))
    itin_no_dest["destination"] = ""
    issues = _run_deterministic_checks(
        request=sample_request,
        draft_itinerary=itin_no_dest,
        research=sample_research,
    )
    assert any("Itinerary destination is missing" in i for i in issues)

    # 3. Missing request currency
    req_no_curr = dict(sample_request)
    req_no_curr["currency"] = ""
    issues = _run_deterministic_checks(
        request=req_no_curr,
        draft_itinerary=valid_itinerary,
        research=sample_research,
    )
    assert any("Travel request currency is missing" in i for i in issues)

    # 4. Missing itinerary currency
    itin_no_curr = json.loads(json.dumps(valid_itinerary))
    itin_no_curr["currency"] = ""
    issues = _run_deterministic_checks(
        request=sample_request,
        draft_itinerary=itin_no_curr,
        research=sample_research,
    )
    assert any("Itinerary currency is missing" in i for i in issues)


def test_validator_detects_malformed_day_dates(
    sample_request, sample_research, valid_itinerary
):
    """Malformed or invalid ISO date values in itinerary days must be reported safely."""
    itin = json.loads(json.dumps(valid_itinerary))
    itin["days"][0]["date"] = "not-a-valid-date"

    issues = _run_deterministic_checks(
        request=sample_request,
        draft_itinerary=itin,
        research=sample_research,
    )

    assert any("Day 1 has malformed or missing date: 'not-a-valid-date'" in i for i in issues)


def test_validator_detects_non_finite_and_boolean_monetary_values(
    sample_request, sample_research, valid_itinerary
):
    """Non-finite floats (NaN, Inf) and boolean values must be rejected in budget and costs."""
    # 1. Non-finite budget
    req_bad_budget = dict(sample_request)
    req_bad_budget["total_trip_budget"] = float("inf")
    issues = _run_deterministic_checks(
        request=req_bad_budget,
        draft_itinerary=valid_itinerary,
        research=sample_research,
    )
    assert any("Invalid total trip budget 'inf'" in i for i in issues)

    # 2. Boolean in cost
    itin_bool_cost = json.loads(json.dumps(valid_itinerary))
    itin_bool_cost["days"][0]["activities"][0]["estimated_cost"] = True
    issues = _run_deterministic_checks(
        request=sample_request,
        draft_itinerary=itin_bool_cost,
        research=sample_research,
    )
    assert any("has invalid estimated_cost 'True'" in i for i in issues)

    # 3. NaN in total_estimated_cost
    itin_nan_cost = json.loads(json.dumps(valid_itinerary))
    itin_nan_cost["total_estimated_cost"] = float("nan")
    issues = _run_deterministic_checks(
        request=sample_request,
        draft_itinerary=itin_nan_cost,
        research=sample_research,
    )
    assert any("Invalid total estimated cost 'nan'" in i for i in issues)
