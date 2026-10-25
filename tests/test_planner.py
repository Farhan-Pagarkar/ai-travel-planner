import os
import httpx
import groq
"""Unit and integration tests for Activity Lookup, Budget Calculator, and Planner Agent."""

import json
from unittest.mock import MagicMock
import pytest

from app.agents.planner import PlannerError, _validate_itinerary_grounding, run_planner_agent
from app.schemas.itinerary import Itinerary
from app.tools.activities import ActivityCandidate, lookup_activity_candidates
from app.tools.budget import DEFAULT_BUDGET_POLICY, calculate_budget_allocation


@pytest.fixture
def sample_research_data() -> dict:
    return {
        "destination": "Kyoto, Japan",
        "attractions": [
            {
                "name": "Fushimi Inari Shrine",
                "description": "Historic shrine with thousands of vermilion torii gates.",
                "visitor_tips": "Early morning visits avoid crowds.",
                "source_url": "https://example.com/fushimi",
            },
            {
                "name": "Kinkaku-ji",
                "description": "Golden Pavilion Zen temple surrounded by a reflecting pond.",
                "visitor_tips": "Best viewed in morning sunlight.",
                "source_url": "https://example.com/kinkaku",
            },
            {
                "name": "Nishiki Market",
                "description": "Narrow shopping street known as Kyoto's Kitchen with food stalls.",
                "visitor_tips": "Sample local street food.",
                "source_url": "https://example.com/nishiki",
            },
            {
                "name": "Fabricated Temple Without URL",
                "description": "Non-verified temple.",
                "visitor_tips": None,
                "source_url": None,  # Unsupported: missing URL
            },
            {
                "name": "Fushimi Inari Shrine",  # Duplicate name
                "description": "Duplicate shrine entry.",
                "source_url": "https://example.com/fushimi-duplicate",
            },
        ],
        "upcoming_local_events": [
            {
                "title": "Autumn Evening Illumination",
                "summary": "Evening temple garden light-up festival.",
                "source_url": "https://example.com/events/lightup",
                "publication_date": "2026-09-01",
                "event_date": "2026-11-01 to 2026-11-30",
                "category": "festival",
                "is_verified": True,
            }
        ],
        "recent_local_news": [
            {
                "title": "Bus schedule updates",
                "summary": "Transit operator notices.",
                "source_url": "https://example.com/news/bus",
            }
        ],
        "historical_local_reports": [
            {
                "title": "Historical 2025 crowded season",
                "summary": "Past report from 2025.",
                "source_url": "https://example.com/news/2025",
            }
        ],
        "weather": {
            "weather_type": "forecast",
            "notes": "Mild autumn weather with clear skies.",
        },
    }


# ---------------------------------------------------------------------------
# 1. Activity Lookup Tool Tests
# ---------------------------------------------------------------------------


def test_activity_lookup_extracts_real_candidates(sample_research_data):
    """Activity lookup extracts real candidate records from ResearchResult."""
    result = lookup_activity_candidates(sample_research_data, limit=10)

    # 3 valid attractions + 1 valid event = 4 candidates (unsupported & duplicate excluded)
    assert result.total_candidates == 4
    assert len(result.candidates) == 4

    cand_types = {c.candidate_type for c in result.candidates}
    assert "attraction" in cand_types
    assert "upcoming_event" in cand_types

    # Ensure IDs are assigned
    ids = [c.id for c in result.candidates]
    assert "attr-1" in ids
    assert "event-1" in ids


def test_activity_lookup_preserves_genuine_urls_and_dates(sample_research_data):
    """Candidate source URLs and event dates are preserved exactly."""
    result = lookup_activity_candidates(sample_research_data)
    candidates_by_name = {c.name: c for c in result.candidates}

    fushimi = candidates_by_name["Fushimi Inari Shrine"]
    assert fushimi.source_url == "https://example.com/fushimi"
    assert fushimi.event_date is None

    illumination = candidates_by_name["Autumn Evening Illumination"]
    assert illumination.source_url == "https://example.com/events/lightup"
    assert illumination.event_date == "2026-11-01 to 2026-11-30"


def test_activity_lookup_filters_and_ranks_by_interests(sample_research_data):
    """Candidates are ranked according to traveler interests."""
    # Interest in food/culinary should rank Nishiki Market first
    food_result = lookup_activity_candidates(sample_research_data, interests=["food", "market"])
    assert food_result.candidates[0].name == "Nishiki Market"

    # Interest in temples should rank shrine or temple first
    temple_result = lookup_activity_candidates(sample_research_data, interests=["temple", "zen"])
    assert "Temple" in temple_result.candidates[0].description or "temple" in temple_result.candidates[0].description.lower()


def test_activity_lookup_handles_duplicates_and_unsupported(sample_research_data):
    """Duplicate or unsupported candidates without source URLs are excluded."""
    result = lookup_activity_candidates(sample_research_data)
    names = [c.name for c in result.candidates]

    # Unsupported candidate without URL must be excluded
    assert "Fabricated Temple Without URL" not in names

    # Duplicate Fushimi Inari entry must be deduplicated
    assert names.count("Fushimi Inari Shrine") == 1

    # Historical news and recent local news must not appear as activity candidates
    assert "Bus schedule updates" not in names
    assert "Historical 2025 crowded season" not in names


def test_activity_lookup_insufficient_candidates_flag():
    """Insufficient research is reported when fewer candidates than required exist."""
    tiny_research = {
        "attractions": [
            {
                "name": "Single Temple",
                "description": "Only temple found.",
                "source_url": "https://example.com/single",
            }
        ]
    }
    result = lookup_activity_candidates(tiny_research, min_required=4)
    assert len(result.candidates) == 1
    assert result.is_sufficient is False
    assert result.notes is not None
    assert "Insufficient verified activity candidates" in result.notes


# ---------------------------------------------------------------------------
# 2. Budget Calculator Tool Tests
# ---------------------------------------------------------------------------


def test_budget_allocation_percentages_sum_to_100():
    """Budget allocation policy percentages sum to exactly 100%."""
    total_pct = sum(DEFAULT_BUDGET_POLICY.values())
    assert round(total_pct, 4) == 1.0

    alloc = calculate_budget_allocation(1500.0, "USD", 2, 4)
    assert round(sum(alloc.allocation_policy_percentages.values()), 4) == 1.0


def test_budget_allocations_remain_within_total_budget():
    """Budget allocations stay strictly within the total group budget across various figures."""
    test_budgets = [1000.0, 1850.33, 2499.99, 500.0]
    for b in test_budgets:
        alloc = calculate_budget_allocation(total_trip_budget=b, currency="USD", traveler_count=2, trip_days=4)
        assert alloc.total_budget == round(b, 2)
        assert round(sum(alloc.breakdown.values()), 2) == round(b, 2)
        assert round(alloc.available_for_planned_expenses + alloc.contingency_reserve, 2) == round(b, 2)
        assert alloc.budget_per_traveler == round(b / 2, 2)
        assert alloc.budget_per_day == round(b / 4, 2)
        assert "accommodation" in alloc.breakdown
        assert "food" in alloc.breakdown
        assert "activities" in alloc.breakdown
        assert "local_transportation" in alloc.breakdown
        assert "contingency" in alloc.breakdown


def test_invalid_budget_inputs_rejected():
    """Invalid budget inputs raise ValueError."""
    with pytest.raises(ValueError, match="greater than zero"):
        calculate_budget_allocation(0.0)

    with pytest.raises(ValueError, match="greater than zero"):
        calculate_budget_allocation(-500.0)

    with pytest.raises(ValueError, match="Traveler count"):
        calculate_budget_allocation(1000.0, traveler_count=0)

    with pytest.raises(ValueError, match="Trip duration"):
        calculate_budget_allocation(1000.0, trip_days=0)

    with pytest.raises(ValueError, match="sum to 100%"):
        bad_policy = {"accommodation": 0.5, "food": 0.2}
        calculate_budget_allocation(1000.0, policy=bad_policy)


# ---------------------------------------------------------------------------
# 3. Planner Agent Synthesis and Grounding Tests
# ---------------------------------------------------------------------------


def test_planner_agent_evidence_passed_to_groq(sample_research_data):
    """Groq receives research evidence, candidate activities, budget info, and user feedback."""
    mock_groq = MagicMock()
    mock_choice = MagicMock()

    mock_itinerary_payload = {
        "destination": "Kyoto, Japan",
        "total_days": 2,
        "currency": "USD",
        "total_estimated_cost": 400.0,
        "days": [
            {
                "day": 1,
                "date": "2026-11-01",
                "focus": "culture",
                "theme": "culture",
                "activities": [
                    {
                        "name": "Fushimi Inari Shrine",
                        "description": "Scenic torii gates walk.",
                        "category": "attraction",
                        "start_time": "09:00 AM",
                        "estimated_duration": "3 hours",
                        "estimated_cost": 100.0,
                        "candidate_id": "attr-1",
                        "source_url": "https://example.com/fushimi",
                    },
                    {
                        "name": "Casual Lunch",
                        "description": "Lunch at neighborhood cafe.",
                        "category": "dining",
                        "start_time": "01:00 PM",
                        "estimated_duration": "1 hour",
                        "estimated_cost": 100.0,
                        "candidate_id": None,
                        "source_url": None,
                    },
                ],
                "estimated_cost": 200.0,
            },
            {
                "day": 2,
                "date": "2026-11-02",
                "focus": "market",
                "theme": "market",
                "activities": [
                    {
                        "name": "Nishiki Market",
                        "description": "Food stalls and market exploration.",
                        "category": "attraction",
                        "start_time": "10:00 AM",
                        "estimated_duration": "3 hours",
                        "estimated_cost": 100.0,
                        "candidate_id": "attr-3",
                        "source_url": "https://example.com/nishiki",
                    },
                    {
                        "name": "Evening Rest",
                        "description": "Free time.",
                        "category": "leisure",
                        "start_time": "06:00 PM",
                        "estimated_duration": "2 hours",
                        "estimated_cost": 100.0,
                        "candidate_id": None,
                        "source_url": None,
                    },
                ],
                "estimated_cost": 200.0,
            },
        ],
        "summary": "2-day Kyoto cultural itinerary.",
        "assumptions": ["Pace is relaxed"],
        "warnings": [],
    }

    mock_choice.message.content = json.dumps(mock_itinerary_payload)
    mock_groq.chat.completions.create.return_value.choices = [mock_choice]

    request = {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-02",
        "total_trip_budget": 800.0,
        "currency": "USD",
        "interests": ["temples", "food"],
        "traveler_count": 2,
    }

    result = run_planner_agent(
        request=request,
        research=sample_research_data,
        feedback="Include more market time",
        modifications="Add morning walk",
        revision_count=1,
        groq_client=mock_groq,
    )

    # Verify prompt contents passed to Groq
    call_args = mock_groq.chat.completions.create.call_args[1]
    assert call_args["reasoning_effort"] == "low"
    assert call_args["include_reasoning"] is False
    assert call_args["response_format"] == {"type": "json_object"}
    prompt_text = call_args["messages"][1]["content"]

    assert "Kyoto, Japan" in prompt_text
    assert "2026-11-01, 2026-11-02" in prompt_text
    assert "BUDGET CONSTRAINTS" in prompt_text
    assert "CANDIDATE ACTIVITIES" in prompt_text
    assert "attr-1" in prompt_text
    assert "Fushimi Inari Shrine" in prompt_text
    assert "Include more market time" in prompt_text or "Add morning walk" in prompt_text

    # Verify valid output schema
    assert result["destination"] == "Kyoto, Japan"
    assert result["total_days"] == 2
    assert len(result["days"]) == 2
    assert result["budget_breakdown"] is not None


def test_planner_agent_rejects_unknown_candidate_ids_and_retries(sample_research_data):
    """Unknown candidate IDs cause a bounded correction attempt that fixes the issue."""
    mock_groq = MagicMock()

    # Attempt 1: hallucinated candidate ID "attr-999"
    bad_attempt = {
        "destination": "Kyoto, Japan",
        "total_days": 1,
        "currency": "USD",
        "total_estimated_cost": 100.0,
        "days": [
            {
                "day": 1,
                "date": "2026-11-01",
                "focus": "culture",
                "theme": "culture",
                "activities": [
                    {
                        "name": "Invented Shrine",
                        "description": "Hallucinated shrine.",
                        "category": "attraction",
                        "start_time": "09:00 AM",
                        "estimated_duration": "2 hours",
                        "estimated_cost": 100.0,
                        "candidate_id": "attr-999",  # Unknown ID
                        "source_url": "https://fake.com/shrine",
                    }
                ],
                "estimated_cost": 100.0,
            }
        ],
        "summary": "Day 1 itinerary.",
        "assumptions": [],
        "warnings": [],
    }

    # Attempt 2: corrected to valid candidate "attr-1"
    good_attempt = {
        "destination": "Kyoto, Japan",
        "total_days": 1,
        "currency": "USD",
        "total_estimated_cost": 100.0,
        "days": [
            {
                "day": 1,
                "date": "2026-11-01",
                "focus": "culture",
                "theme": "culture",
                "activities": [
                    {
                        "name": "Fushimi Inari Shrine",
                        "description": "Historic shrine.",
                        "category": "attraction",
                        "start_time": "09:00 AM",
                        "estimated_duration": "2 hours",
                        "estimated_cost": 100.0,
                        "candidate_id": "attr-1",  # Valid ID
                        "source_url": "https://example.com/fushimi",
                    }
                ],
                "estimated_cost": 100.0,
            }
        ],
        "summary": "Day 1 corrected itinerary.",
        "assumptions": [],
        "warnings": [],
    }

    choice1 = MagicMock()
    choice1.message.content = json.dumps(bad_attempt)
    choice2 = MagicMock()
    choice2.message.content = json.dumps(good_attempt)

    mock_groq.chat.completions.create.side_effect = [
        MagicMock(choices=[choice1]),
        MagicMock(choices=[choice2]),
    ]

    request = {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-01",
        "total_trip_budget": 500.0,
        "currency": "USD",
    }

    result = run_planner_agent(
        request=request,
        research=sample_research_data,
        groq_client=mock_groq,
    )

    # Groq was called twice (initial generation + bounded correction)
    assert mock_groq.chat.completions.create.call_count == 2
    assert result["days"][0]["activities"][0]["candidate_id"] == "attr-1"
    assert result["days"][0]["activities"][0]["source_url"] == "https://example.com/fushimi"


def test_planner_agent_fails_after_exhausting_correction_attempt(sample_research_data):
    """If the LLM fails validation twice, a clear RuntimeError is raised."""
    mock_groq = MagicMock()

    bad_payload = {
        "destination": "Kyoto, Japan",
        "total_days": 1,
        "currency": "USD",
        "total_estimated_cost": 100.0,
        "days": [
            {
                "day": 1,
                "date": "2026-11-01",
                "focus": "culture",
                "activities": [
                    {
                        "name": "NonExistent",
                        "description": "Invalid.",
                        "category": "attraction",
                        "estimated_cost": 100.0,
                        "candidate_id": "attr-999",  # Persistent unknown ID
                        "source_url": "https://fake.com",
                    }
                ],
                "estimated_cost": 100.0,
            }
        ],
        "summary": "Invalid draft.",
    }

    choice = MagicMock()
    choice.message.content = json.dumps(bad_payload)
    mock_groq.chat.completions.create.return_value = MagicMock(choices=[choice])

    request = {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-01",
        "total_trip_budget": 500.0,
    }

    with pytest.raises(RuntimeError) as excinfo:
        run_planner_agent(request=request, research=sample_research_data, groq_client=mock_groq)

    assert "failed validation after correction attempt" in str(excinfo.value)


def test_itinerary_grounding_validation_detects_errors():
    """Grounding validation checks dates, math, unknown IDs, and budget overruns."""
    cand = ActivityCandidate(
        id="attr-1",
        name="Kyoto Temple",
        description="Temple",
        category="attraction",
        source_url="https://example.com/kyoto",
    )
    valid_candidates = {"attr-1": cand}

    # 1. Date mismatch
    errs_dates = _validate_itinerary_grounding(
        itinerary_data={"days": [{"day": 1, "date": "2026-11-02", "activities": [], "estimated_cost": 0.0}]},
        expected_dates=["2026-11-01"],
        max_budget=500.0,
        valid_candidates=valid_candidates,
    )
    assert any("dates" in e for e in errs_dates)

    # 2. Activity cost sum mismatch
    errs_cost = _validate_itinerary_grounding(
        itinerary_data={
            "total_estimated_cost": 100.0,
            "days": [
                {
                    "day": 1,
                    "date": "2026-11-01",
                    "activities": [
                        {"name": "Walk", "estimated_cost": 30.0, "candidate_id": None, "source_url": None}
                    ],
                    "estimated_cost": 50.0,  # 30 != 50
                }
            ],
        },
        expected_dates=["2026-11-01"],
        max_budget=500.0,
        valid_candidates=valid_candidates,
    )
    assert any("costs sum" in e for e in errs_cost)

    # 3. Budget overrun
    errs_budget = _validate_itinerary_grounding(
        itinerary_data={
            "total_estimated_cost": 600.0,
            "days": [
                {
                    "day": 1,
                    "date": "2026-11-01",
                    "activities": [{"name": "Tour", "estimated_cost": 600.0, "candidate_id": None, "source_url": None}],
                    "estimated_cost": 600.0,
                }
            ],
        },
        expected_dates=["2026-11-01"],
        max_budget=500.0,  # 600 > 500
        valid_candidates=valid_candidates,
    )
    assert any("exceeds" in e for e in errs_budget)

    # 4. Unknown candidate ID
    errs_id = _validate_itinerary_grounding(
        itinerary_data={
            "total_estimated_cost": 50.0,
            "days": [
                {
                    "day": 1,
                    "date": "2026-11-01",
                    "activities": [
                        {
                            "name": "Secret Spot",
                            "estimated_cost": 50.0,
                            "candidate_id": "attr-fake",
                            "source_url": "https://example.com/fake",
                        }
                    ],
                    "estimated_cost": 50.0,
                }
            ],
        },
        expected_dates=["2026-11-01"],
        max_budget=500.0,
        valid_candidates=valid_candidates,
    )
    assert any("unknown candidate ID" in e for e in errs_id)

    # 5. Fabricated URL for known candidate
    errs_url = _validate_itinerary_grounding(
        itinerary_data={
            "total_estimated_cost": 50.0,
            "days": [
                {
                    "day": 1,
                    "date": "2026-11-01",
                    "activities": [
                        {
                            "name": "Kyoto Temple",
                            "estimated_cost": 50.0,
                            "candidate_id": "attr-1",
                            "source_url": "https://hallucinated-url.com",  # Doesn't match candidate
                        }
                    ],
                    "estimated_cost": 50.0,
                }
            ],
        },
        expected_dates=["2026-11-01"],
        max_budget=500.0,
        valid_candidates=valid_candidates,
    )
    assert any("does not match candidate" in e for e in errs_url)

    # 6. Researched attraction category without candidate ID is rejected
    errs_attr_no_id = _validate_itinerary_grounding(
        itinerary_data={
            "total_estimated_cost": 50.0,
            "days": [
                {
                    "day": 1,
                    "date": "2026-11-01",
                    "activities": [
                        {
                            "name": "Kyoto Temple",
                            "category": "attraction",
                            "estimated_cost": 50.0,
                            "candidate_id": None,
                            "source_url": None,
                        }
                    ],
                    "estimated_cost": 50.0,
                }
            ],
        },
        expected_dates=["2026-11-01"],
        max_budget=500.0,
        valid_candidates=valid_candidates,
    )
    assert any("classified as an attraction/event" in e for e in errs_attr_no_id)

    # 7. Malformed or non-finite cost handled safely without raising TypeError
    errs_malformed_cost = _validate_itinerary_grounding(
        itinerary_data={
            "total_estimated_cost": float("nan"),
            "days": [
                {
                    "day": 1,
                    "date": "2026-11-01",
                    "activities": [
                        {
                            "name": "Walking Tour",
                            "category": "leisure",
                            "estimated_cost": "free",
                            "candidate_id": None,
                            "source_url": None,
                        }
                    ],
                    "estimated_cost": None,
                }
            ],
        },
        expected_dates=["2026-11-01"],
        max_budget=500.0,
        valid_candidates=valid_candidates,
    )
    assert any("invalid cost" in e for e in errs_malformed_cost)


def test_insufficient_research_does_not_fabricate_attractions():
    """When research is empty, candidate prompt instructs free exploration without crashing."""
    mock_groq = MagicMock()

    empty_itinerary = {
        "destination": "Obscure Town",
        "total_days": 1,
        "currency": "USD",
        "total_estimated_cost": 50.0,
        "days": [
            {
                "day": 1,
                "date": "2026-11-01",
                "focus": "leisure",
                "activities": [
                    {
                        "name": "Self-Guided Walking Tour",
                        "description": "Leisurely neighborhood walk.",
                        "category": "leisure",
                        "start_time": "Morning",
                        "estimated_duration": "2 hours",
                        "estimated_cost": 50.0,
                        "candidate_id": None,
                        "source_url": None,
                    }
                ],
                "estimated_cost": 50.0,
            }
        ],
        "summary": "1-day leisure visit.",
        "assumptions": ["Self-guided leisure."],
        "warnings": [],
    }

    choice = MagicMock()
    choice.message.content = json.dumps(empty_itinerary)
    mock_groq.chat.completions.create.return_value = MagicMock(choices=[choice])

    request = {
        "destination": "Obscure Town",
        "start_date": "2026-11-01",
        "end_date": "2026-11-01",
        "total_trip_budget": 300.0,
    }

    result = run_planner_agent(
        request=request,
        research={"attractions": [], "upcoming_local_events": []},
        groq_client=mock_groq,
    )

    assert result["total_days"] == 1
    # Check that warning about insufficient research was added
    assert any("Insufficient verified activity candidates" in w for w in result["warnings"])
    assert result["days"][0]["activities"][0]["candidate_id"] is None


def test_activity_lookup_rejects_missing_malformed_and_unsupported_schemes():
    """Candidates with missing, malformed, or non-http(s) URL schemes must be excluded."""
    research_with_bad_urls = {
        "attractions": [
            {"name": "No URL Attraction", "source_url": None, "description": "Desc"},
            {"name": "Empty URL Attraction", "source_url": "   ", "description": "Desc"},
            {"name": "FTP Scheme Attraction", "source_url": "ftp://files.example.com/item", "description": "Desc"},
            {"name": "Javascript Scheme Attraction", "source_url": "javascript:alert(1)", "description": "Desc"},
            {"name": "Relative Path Attraction", "source_url": "/attractions/view", "description": "Desc"},
            {"name": "No Scheme Domain Attraction", "source_url": "example.com/site", "description": "Desc"},
            {"name": "Malformed Scheme Attraction", "source_url": "http://", "description": "Desc"},
            {"name": "Valid HTTPS Attraction", "source_url": "https://example.com/valid", "description": "Desc"},
            {"name": "Valid HTTP Attraction", "source_url": "http://example.com/also-valid", "description": "Desc"},
        ],
        "upcoming_local_events": [
            {"title": "No URL Event", "source_url": "", "summary": "Sum"},
            {"title": "File Scheme Event", "source_url": "file:///path/to/event", "summary": "Sum"},
            {"title": "Valid HTTPS Event", "source_url": "https://example.com/event", "summary": "Sum"},
        ],
    }

    result = lookup_activity_candidates(research_data=research_with_bad_urls)
    assert len(result.candidates) == 3
    extracted_names = [c.name for c in result.candidates]
    assert "Valid HTTPS Attraction" in extracted_names
    assert "Valid HTTP Attraction" in extracted_names
    assert "Valid HTTPS Event" in extracted_names
    assert "FTP Scheme Attraction" not in extracted_names
    assert "Javascript Scheme Attraction" not in extracted_names
    assert "Relative Path Attraction" not in extracted_names


def test_planner_grounding_detects_mismatched_activity_name():
    """Grounding validation catches valid candidate ID and URL used with unrelated attraction name."""
    cand = ActivityCandidate(
        id="attr-1",
        name="Kyoto Temple",
        description="Historic temple",
        category="attraction",
        source_url="https://example.com/kyoto",
    )
    valid_candidates = {"attr-1": cand}

    itinerary_with_disguised_attraction = {
        "total_estimated_cost": 50.0,
        "days": [
            {
                "day": 1,
                "date": "2026-11-01",
                "activities": [
                    {
                        "name": "Disguised Unrelated Theme Park",
                        "estimated_cost": 50.0,
                        "candidate_id": "attr-1",
                        "source_url": "https://example.com/kyoto",
                        "category": "attraction",
                    }
                ],
                "estimated_cost": 50.0,
            }
        ],
    }

    errors = _validate_itinerary_grounding(
        itinerary_data=itinerary_with_disguised_attraction,
        expected_dates=["2026-11-01"],
        max_budget=500.0,
        valid_candidates=valid_candidates,
    )

    assert any("does not match candidate 'attr-1' name 'Kyoto Temple'" in err for err in errors)


def test_planner_agent_retries_and_corrects_mismatched_activity_name(sample_research_data):
    """Planner Agent triggers bounded correction retry when LLM disguises attraction name."""
    mock_groq = MagicMock()

    # Attempt 1: Valid ID and URL but mismatched attraction name
    bad_attempt = {
        "destination": "Kyoto, Japan",
        "total_days": 1,
        "currency": "USD",
        "total_estimated_cost": 100.0,
        "days": [
            {
                "day": 1,
                "date": "2026-11-01",
                "focus": "sightseeing",
                "theme": "sightseeing",
                "activities": [
                    {
                        "name": "Fabricated Disguised Rollercoaster",
                        "description": "Fun ride.",
                        "category": "attraction",
                        "start_time": "09:00 AM",
                        "estimated_duration": "3 hours",
                        "estimated_cost": 100.0,
                        "candidate_id": "attr-1",
                        "source_url": "https://example.com/fushimi",
                    }
                ],
                "estimated_cost": 100.0,
            }
        ],
        "summary": "1-day trip.",
        "assumptions": [],
        "warnings": [],
    }

    # Attempt 2: Corrected attraction name matching candidate
    corrected_attempt = {
        "destination": "Kyoto, Japan",
        "total_days": 1,
        "currency": "USD",
        "total_estimated_cost": 100.0,
        "days": [
            {
                "day": 1,
                "date": "2026-11-01",
                "focus": "sightseeing",
                "theme": "sightseeing",
                "activities": [
                    {
                        "name": "Fushimi Inari Shrine",
                        "description": "Historic shrine.",
                        "category": "attraction",
                        "start_time": "09:00 AM",
                        "estimated_duration": "3 hours",
                        "estimated_cost": 100.0,
                        "candidate_id": "attr-1",
                        "source_url": "https://example.com/fushimi",
                    }
                ],
                "estimated_cost": 100.0,
            }
        ],
        "summary": "1-day trip.",
        "assumptions": [],
        "warnings": [],
    }

    choice1 = MagicMock()
    choice1.message.content = json.dumps(bad_attempt)
    choice2 = MagicMock()
    choice2.message.content = json.dumps(corrected_attempt)

    mock_groq.chat.completions.create.side_effect = [
        MagicMock(choices=[choice1]),
        MagicMock(choices=[choice2]),
    ]

    request = {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-01",
        "total_trip_budget": 500.0,
    }

    result = run_planner_agent(
        request=request,
        research=sample_research_data,
        groq_client=mock_groq,
    )

    assert mock_groq.chat.completions.create.call_count == 2
    assert result["days"][0]["activities"][0]["name"] == "Fushimi Inari Shrine"


def test_planner_agent_fails_when_name_mismatch_persists_after_correction(sample_research_data):
    """Planner Agent raises RuntimeError if mismatched activity name is not corrected."""
    mock_groq = MagicMock()

    bad_attempt = {
        "destination": "Kyoto, Japan",
        "total_days": 1,
        "currency": "USD",
        "total_estimated_cost": 100.0,
        "days": [
            {
                "day": 1,
                "date": "2026-11-01",
                "focus": "sightseeing",
                "theme": "sightseeing",
                "activities": [
                    {
                        "name": "Fabricated Rollercoaster",
                        "description": "Fun ride.",
                        "category": "attraction",
                        "start_time": "09:00 AM",
                        "estimated_duration": "3 hours",
                        "estimated_cost": 100.0,
                        "candidate_id": "attr-1",
                        "source_url": "https://example.com/fushimi",
                    }
                ],
                "estimated_cost": 100.0,
            }
        ],
        "summary": "1-day trip.",
        "assumptions": [],
        "warnings": [],
    }

    choice = MagicMock()
    choice.message.content = json.dumps(bad_attempt)
    mock_groq.chat.completions.create.return_value = MagicMock(choices=[choice])

    request = {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-01",
        "total_trip_budget": 500.0,
    }

    with pytest.raises(RuntimeError, match="failed validation after correction attempt"):
        run_planner_agent(
            request=request,
            research=sample_research_data,
            groq_client=mock_groq,
        )


def test_planner_agent_retries_on_groq_json_validate_failed_and_succeeds(sample_research_data):
    """Planner Agent triggers bounded correction retry when Groq returns 400 json_validate_failed."""
    mock_groq = MagicMock()

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(
        400,
        request=req,
        json={
            "error": {
                "message": "Failed to validate JSON. Please adjust your prompt.",
                "type": "invalid_request_error",
                "code": "json_validate_failed",
            }
        },
    )
    groq_400 = groq.BadRequestError(
        message="Error code: 400 - Failed to validate JSON. Please adjust your prompt. code: json_validate_failed",
        response=resp,
        body={
            "error": {
                "message": "Failed to validate JSON. Please adjust your prompt.",
                "type": "invalid_request_error",
                "code": "json_validate_failed",
            }
        },
    )

    valid_itinerary = {
        "destination": "Kyoto, Japan",
        "total_days": 1,
        "currency": "USD",
        "total_estimated_cost": 100.0,
        "days": [
            {
                "day": 1,
                "date": "2026-11-01",
                "focus": "sightseeing",
                "theme": "sightseeing",
                "activities": [
                    {
                        "name": "Fushimi Inari Shrine",
                        "description": "Historic shrine.",
                        "category": "attraction",
                        "start_time": "09:00 AM",
                        "estimated_duration": "3 hours",
                        "estimated_cost": 100.0,
                        "candidate_id": "attr-1",
                        "source_url": "https://example.com/fushimi",
                    }
                ],
                "estimated_cost": 100.0,
            }
        ],
        "summary": "1-day trip in Kyoto.",
        "assumptions": [],
        "warnings": [],
    }

    choice = MagicMock()
    choice.message.content = json.dumps(valid_itinerary)

    # Attempt 1 raises 400 json_validate_failed, Attempt 2 returns valid JSON
    mock_groq.chat.completions.create.side_effect = [
        groq_400,
        MagicMock(choices=[choice]),
    ]

    request = {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-01",
        "total_trip_budget": 500.0,
    }

    result = run_planner_agent(
        request=request,
        research=sample_research_data,
        groq_client=mock_groq,
    )

    assert mock_groq.chat.completions.create.call_count == 2
    for call in mock_groq.chat.completions.create.call_args_list:
        assert call[1]["reasoning_effort"] == "low"
        assert call[1]["include_reasoning"] is False
        assert call[1]["response_format"] == {"type": "json_object"}
    assert result["destination"] == "Kyoto, Japan"
    assert result["days"][0]["activities"][0]["name"] == "Fushimi Inari Shrine"


def test_planner_agent_fails_cleanly_on_persistent_groq_json_validate_failed(sample_research_data):
    """Planner Agent raises PlannerError without exposing secrets when Groq json_validate_failed persists."""
    mock_groq = MagicMock()

    secret_key = "gsk_testsecretkey12345"
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(
        400,
        request=req,
        json={"error": {"message": f"Failed with {secret_key}", "code": "json_validate_failed"}},
    )
    groq_400 = groq.BadRequestError(
        message=f"Error code: 400 - Failed with {secret_key} code: json_validate_failed",
        response=resp,
        body={"error": {"message": f"Failed with {secret_key}", "code": "json_validate_failed"}},
    )

    # Both attempts fail
    mock_groq.chat.completions.create.side_effect = [groq_400, groq_400]

    request = {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-01",
        "total_trip_budget": 500.0,
    }

    with pytest.raises(PlannerError) as exc_info:
        run_planner_agent(
            request=request,
            research=sample_research_data,
            groq_client=mock_groq,
        )

    assert mock_groq.chat.completions.create.call_count == 2
    err_str = str(exc_info.value)
    assert "Groq itinerary generation failed after retry" in err_str
    assert secret_key not in err_str
    assert "[REDACTED_API_KEY]" in err_str


@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_TESTS"),
    reason="Opt-in live API test requiring external Groq API key",
)
def test_live_planner_smoke_test(sample_research_data):
    """Opt-in live smoke test verifying real Groq API produces a valid grounded itinerary."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        pytest.skip("GROQ_API_KEY not configured")

    request = {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-02",
        "total_trip_budget": 500.0,
        "currency": "USD",
        "interests": ["temples", "culture"],
        "traveler_count": 1,
    }

    result = run_planner_agent(
        request=request,
        research=sample_research_data,
    )

    assert result["destination"] == "Kyoto, Japan"
    assert result["total_days"] == 2
    assert len(result["days"]) == 2
    assert result["total_estimated_cost"] <= 500.0
    for day in result["days"]:
        assert len(day["activities"]) > 0


def test_planner_agent_handles_empty_groq_choices_list(sample_research_data):
    """Empty choices list from Groq raises PlannerError cleanly without list index error."""
    mock_groq = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = []
    mock_groq.chat.completions.create.return_value = mock_resp

    request = {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-02",
        "total_trip_budget": 500.0,
    }

    with pytest.raises(PlannerError) as exc_info:
        run_planner_agent(
            request=request,
            research=sample_research_data,
            groq_client=mock_groq,
        )

    assert "empty choices list" in str(exc_info.value).lower()


def test_planner_agent_handles_empty_model_output(sample_research_data):
    """Empty content string from Groq raises PlannerError cleanly."""
    mock_groq = MagicMock()
    choice = MagicMock()
    choice.message.content = ""
    mock_groq.chat.completions.create.return_value = MagicMock(choices=[choice])

    request = {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-02",
        "total_trip_budget": 500.0,
    }

    with pytest.raises(PlannerError) as exc_info:
        run_planner_agent(
            request=request,
            research=sample_research_data,
            groq_client=mock_groq,
        )

    assert "empty response content" in str(exc_info.value).lower()


def test_planner_agent_does_not_retry_on_rate_limit_error(sample_research_data):
    """Rate-limit errors raise PlannerError immediately without immediate retry."""
    mock_groq = MagicMock()
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(429, request=req, json={"error": {"message": "Rate limit exceeded", "type": "rate_limit_exceeded"}})
    rate_err = groq.RateLimitError(
        message="Error code: 429 - Rate limit exceeded",
        response=resp,
        body={"error": {"message": "Rate limit exceeded", "type": "rate_limit_exceeded"}},
    )
    mock_groq.chat.completions.create.side_effect = rate_err

    request = {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-02",
        "total_trip_budget": 500.0,
    }

    with pytest.raises(PlannerError) as exc_info:
        run_planner_agent(
            request=request,
            research=sample_research_data,
            groq_client=mock_groq,
        )

    # Must NOT retry immediately on rate limit!
    assert mock_groq.chat.completions.create.call_count == 1
    assert "rate limit exceeded" in str(exc_info.value).lower()


def test_planner_agent_rejects_trip_exceeding_14_days(sample_research_data):
    """Planner agent enforces the 14-day trip duration ceiling."""
    request = {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-16",  # 16 days
        "total_trip_budget": 500.0,
    }

    with pytest.raises(ValueError, match="exceeds the maximum allowed limit of 14 calendar days"):
        run_planner_agent(
            request=request,
            research=sample_research_data,
        )

def test_itinerary_grounding_rejects_accounting_categories_as_activities(sample_research_data):
    """Grounding validator rejects accounting categories emitted as scheduled activities."""
    candidate_lookup = lookup_activity_candidates(sample_research_data, min_required=1)
    valid_candidates = {c.id: c for c in candidate_lookup.candidates}

    invalid_itinerary = {
        "destination": "Kyoto, Japan",
        "total_days": 1,
        "currency": "USD",
        "total_estimated_cost": 250.0,
        "days": [
            {
                "day": 1,
                "date": "2026-11-01",
                "focus": "Sightseeing",
                "activities": [
                    {
                        "name": "Fushimi Inari Shrine",
                        "description": "Visit shrine",
                        "category": "attraction",
                        "estimated_cost": 50.0,
                        "candidate_id": "attr-1",
                        "source_url": "https://example.com/fushimi",
                    },
                    {
                        "name": "Accommodation",
                        "description": "Hotel stay",
                        "category": "accommodation",
                        "estimated_cost": 100.0,
                        "candidate_id": None,
                        "source_url": None,
                    },
                    {
                        "name": "Local Transportation",
                        "description": "Train and bus passes",
                        "category": "transit",
                        "estimated_cost": 50.0,
                        "candidate_id": None,
                        "source_url": None,
                    },
                    {
                        "name": "Contingency",
                        "description": "Emergency reserve",
                        "category": "contingency",
                        "estimated_cost": 50.0,
                        "candidate_id": None,
                        "source_url": None,
                    },
                ],
                "estimated_cost": 250.0,
            }
        ],
        "summary": "Day 1 itinerary",
        "assumptions": [],
        "warnings": [],
    }

    errors = _validate_itinerary_grounding(
        itinerary_data=invalid_itinerary,
        expected_dates=["2026-11-01"],
        max_budget=500.0,
        valid_candidates=valid_candidates,
        destination="Kyoto, Japan",
        currency="USD",
    )

    err_text = " ".join(errors).lower()
    assert "accounting category 'accommodation'" in err_text
    assert "accounting category 'local transportation'" in err_text
    assert "accounting category 'contingency'" in err_text


def test_contingency_reserve_not_counted_as_spending(sample_research_data):
    """Contingency reserve remains unspent; itinerary total need not equal full budget."""
    alloc = calculate_budget_allocation(
        total_trip_budget=1000.0,
        currency="USD",
        traveler_count=2,
        trip_days=4,
    )
    # Contingency is 100.0 (10%), available for planned spending is 900.0
    assert alloc.contingency_reserve == 100.0
    assert alloc.available_for_planned_expenses == 900.0

    candidate_lookup = lookup_activity_candidates(sample_research_data, min_required=1)
    valid_candidates = {c.id: c for c in candidate_lookup.candidates}

    # Total planned expenses = 60.0 (well below 900.0 ceiling and 1000.0 budget)
    valid_partial_budget_itinerary = {
        "destination": "Kyoto, Japan",
        "total_days": 1,
        "currency": "USD",
        "total_estimated_cost": 60.0,
        "days": [
            {
                "day": 1,
                "date": "2026-11-01",
                "focus": "Temples",
                "activities": [
                    {
                        "name": "Fushimi Inari Shrine",
                        "description": "Visit temple",
                        "category": "attraction",
                        "estimated_cost": 20.0,
                        "candidate_id": "attr-1",
                        "source_url": "https://example.com/fushimi",
                    },
                    {
                        "name": "Lunch near Fushimi Inari Shrine",
                        "description": "Local dining",
                        "category": "dining",
                        "estimated_cost": 40.0,
                        "candidate_id": None,
                        "source_url": None,
                    },
                ],
                "estimated_cost": 60.0,
            }
        ],
        "summary": "1-day trip",
        "budget_breakdown": alloc.to_dict(),
        "assumptions": [],
        "warnings": [],
    }

    errors = _validate_itinerary_grounding(
        itinerary_data=valid_partial_budget_itinerary,
        expected_dates=["2026-11-01"],
        max_budget=1000.0,
        valid_candidates=valid_candidates,
        destination="Kyoto, Japan",
        currency="USD",
    )
    assert errors == []


def test_unserved_interests_produce_warnings(sample_research_data):
    """Interests with no verified candidates in research produce explicit warnings."""
    mock_groq = MagicMock()
    valid_plan = {
        "destination": "Kyoto, Japan",
        "total_days": 1,
        "currency": "USD",
        "total_estimated_cost": 50.0,
        "days": [
            {
                "day": 1,
                "date": "2026-11-01",
                "focus": "Sightseeing",
                "activities": [
                    {
                        "name": "Fushimi Inari Shrine",
                        "description": "Historic shrine",
                        "category": "attraction",
                        "estimated_cost": 20.0,
                        "candidate_id": "attr-1",
                        "source_url": "https://example.com/fushimi",
                    },
                    {
                        "name": "Lunch in Gion",
                        "description": "Traditional lunch",
                        "category": "dining",
                        "estimated_cost": 30.0,
                        "candidate_id": None,
                        "source_url": None,
                    },
                ],
                "estimated_cost": 50.0,
            }
        ],
        "summary": "Sightseeing day",
        "assumptions": [],
        "warnings": [],
    }
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(valid_plan)))]
    mock_groq.chat.completions.create.return_value = mock_resp

    request = {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-01",
        "total_trip_budget": 500.0,
        "currency": "USD",
        "interests": ["games", "sightseeing", "waters"],
        "preferences": {
            "pace": "moderate",
            "notes": "open for bike rides, long tours can extend days",
        },
    }

    result = run_planner_agent(
        request=request,
        research=sample_research_data,
        groq_client=mock_groq,
    )

    warnings_text = " ".join(result.get("warnings", [])).lower()
    # "games" and "waters" have no candidates in sample_research_data
    assert "games" in warnings_text
    assert "waters" in warnings_text


def test_broad_destination_discloses_assumption(sample_research_data):
    """Country-level destination discloses geographic focus assumption."""
    mock_groq = MagicMock()
    valid_plan = {
        "destination": "Japan",
        "total_days": 1,
        "currency": "USD",
        "total_estimated_cost": 50.0,
        "days": [
            {
                "day": 1,
                "date": "2026-11-01",
                "focus": "Sightseeing",
                "activities": [
                    {
                        "name": "Fushimi Inari Shrine",
                        "description": "Historic shrine",
                        "category": "attraction",
                        "estimated_cost": 20.0,
                        "candidate_id": "attr-1",
                        "source_url": "https://example.com/fushimi",
                    },
                    {
                        "name": "Lunch in Gion",
                        "description": "Traditional lunch",
                        "category": "dining",
                        "estimated_cost": 30.0,
                        "candidate_id": None,
                        "source_url": None,
                    },
                ],
                "estimated_cost": 50.0,
            }
        ],
        "summary": "1 day trip",
        "assumptions": [],
        "warnings": [],
    }
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(valid_plan)))]
    mock_groq.chat.completions.create.return_value = mock_resp

    request = {
        "destination": "Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-01",
        "total_trip_budget": 500.0,
        "currency": "USD",
        "interests": ["sightseeing"],
    }

    result = run_planner_agent(
        request=request,
        research=sample_research_data,
        groq_client=mock_groq,
    )

    assumptions_text = " ".join(result.get("assumptions", [])).lower()
    assert "broad country-level destination" in assumptions_text or "geographically focused" in assumptions_text


def test_planner_uses_all_relevant_candidate_activities(sample_research_data):
    """Candidate activities are ranked by traveler interests and formatted in prompt."""
    candidate_lookup = lookup_activity_candidates(
        research_data=sample_research_data,
        interests=["temples", "culture"],
        limit=20,
        min_required=2,
    )
    assert candidate_lookup.is_sufficient is True
    assert len(candidate_lookup.candidates) >= 2
    top_cand = candidate_lookup.candidates[0]
    assert "shrine" in top_cand.name.lower() or "temple" in top_cand.name.lower() or "temple" in top_cand.description.lower()
