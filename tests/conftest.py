"""Shared pytest fixtures and test boundary mocks."""

from typing import Any
import pytest

MOCK_RESEARCH_RESULT: dict[str, Any] = {
    "destination": "Kyoto, Japan",
    "coordinates": {
        "latitude": 35.02107,
        "longitude": 135.75385,
        "resolved_name": "Kyoto, Japan",
        "timezone": "Asia/Tokyo",
    },
    "attractions": [
        {
            "name": "Fushimi Inari Shrine",
            "description": "Historic shrine renowned for its scenic red torii gate path.",
            "visitor_tips": "Early morning visits avoid crowds.",
            "source_url": "https://example.com/fushimi",
        }
    ],
    "local_tips": ["Purchase an IC transit card for regional bus and rail access."],
    "safety_considerations": ["Destination is very safe; carry hydration for walking tours."],
    "seasonal_considerations": ["Autumn foliage peak season brings pleasant temperatures and high demand."],
    "recent_local_news": [
        {
            "title": "Kyoto city updates bus scheduling for autumn travel",
            "summary": "Regional transport operators announce adjusted weekend schedules.",
            "source_url": "https://example.com/news/bus-schedule",
            "publication_date": "2026-10-20",
            "event_date": None,
            "category": "transport",
            "potential_travel_impact": "low",
            "is_verified": True,
        }
    ],
    "historical_local_reports": [
        {
            "title": "Strong autumn tourism numbers noted in November 2025",
            "summary": "Historic temple areas reported high foot traffic throughout early November 2025.",
            "source_url": "https://example.com/news/autumn-2025",
            "publication_date": "2025-11-15",
            "event_date": None,
            "category": "weather",
            "potential_travel_impact": "moderate",
            "is_verified": True,
        }
    ],
    "upcoming_local_events": [
        {
            "title": "Autumn Evening Temple Light-Up",
            "summary": "Evening illumination events across select temple gardens.",
            "source_url": "https://example.com/events/lightup",
            "publication_date": "2026-09-01",
            "event_date": "2026-11-01 to 2026-11-30",
            "category": "festival",
            "potential_travel_impact": "moderate",
            "is_verified": True,
        }
    ],
    "weather": {
        "weather_type": "historical_comparison",
        "forecast_days": [],
        "historical_comparison": {
            "comparison_years": [2025, 2024, 2023],
            "equivalent_dates_last_year": "2025-11-01 to 2025-11-04",
            "typical_temp_max_c": 19.5,
            "typical_temp_min_c": 9.8,
            "typical_precipitation_mm": 12.0,
            "rainy_days_count": 1,
            "summary": "Mild autumn conditions with daytime highs around 19°C.",
        },
        "notes": "Historical comparison data used for seasonal expectations.",
    },
    "source_urls": [
        "https://example.com/fushimi",
        "https://example.com/news/bus-schedule",
        "https://example.com/news/autumn-2025",
    ],
    "warnings_and_uncertainties": [],
}


@pytest.fixture(autouse=True)
def mock_researcher_in_workflow_tests(request, monkeypatch):
    """Mock the Research Agent during workflow and API integration tests.

    Ensures zero external network calls, zero credit consumption, and complete
    offline test reliability unless explicitly running test_research.py.
    """
    filename = request.node.fspath.basename if hasattr(request.node, "fspath") else ""
    if "test_research" not in filename:
        def fake_research(req: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
            dest = req.get("destination", "Kyoto, Japan")
            res = dict(MOCK_RESEARCH_RESULT)
            res["destination"] = dest
            return res

        monkeypatch.setattr("app.graph.nodes.run_researcher_agent", fake_research)
        monkeypatch.setattr("app.agents.researcher.run_researcher_agent", fake_research)


@pytest.fixture(autouse=True)
def mock_planner_in_workflow_tests(request, monkeypatch):
    """Mock the Planner Agent during workflow and API integration tests.

    Ensures zero external LLM calls and fast offline execution unless
    explicitly running test_planner.py.
    """
    from datetime import date, datetime, timedelta

    filename = request.node.fspath.basename if hasattr(request.node, "fspath") else ""
    if "test_planner" not in filename:
        def fake_planner(
            request: dict[str, Any],
            research: Any = None,
            feedback: Any = None,
            modifications: Any = None,
            revision_count: int = 0,
            *args,
            **kwargs,
        ) -> dict[str, Any]:
            dest = request.get("destination", "Rome, Italy")
            currency = request.get("currency", "USD")
            budget = float(request.get("total_trip_budget", 1000.0))
            interests = request.get("interests", ["sightseeing"])

            try:
                s_dt = datetime.strptime(str(request["start_date"]), "%Y-%m-%d").date()
                e_dt = datetime.strptime(str(request["end_date"]), "%Y-%m-%d").date()
                n_days = max(1, (e_dt - s_dt).days + 1)
            except Exception:
                s_dt = date.today()
                n_days = 4

            target_budget = round(budget * 0.75, 2)
            day_budget = round(target_budget / n_days, 2)

            days = []
            for d_idx in range(n_days):
                curr_date = s_dt + timedelta(days=d_idx)
                interest = interests[d_idx % len(interests)]

                acts = [
                    {
                        "name": f"Morning {interest.title()} Tour",
                        "description": f"Morning visit in {dest} focusing on {interest}.",
                        "category": "attraction",
                        "start_time": "09:00 AM",
                        "estimated_duration": "3 hours",
                        "estimated_cost": round(day_budget * 0.5, 2),
                        "candidate_id": "attr-1",
                        "source_url": "https://example.com/fushimi",
                    },
                    {
                        "name": "Local Dining Experience",
                        "description": "Lunch and culinary exploration in historic quarter.",
                        "category": "dining",
                        "start_time": "01:00 PM",
                        "estimated_duration": "2 hours",
                        "estimated_cost": round(day_budget * 0.3, 2),
                        "candidate_id": None,
                        "source_url": None,
                    },
                    {
                        "name": "Evening Promenade",
                        "description": "Leisure stroll and relaxation.",
                        "category": "leisure",
                        "start_time": "06:00 PM",
                        "estimated_duration": "2 hours",
                        "estimated_cost": round(day_budget - (round(day_budget * 0.5, 2) + round(day_budget * 0.3, 2)), 2),
                        "candidate_id": None,
                        "source_url": None,
                    },
                ]

                if modifications:
                    acts.append({
                        "name": "Requested Modification",
                        "description": f"Modification applied: {modifications}",
                        "category": "leisure",
                        "start_time": "08:00 PM",
                        "estimated_duration": "1 hour",
                        "estimated_cost": 0.0,
                        "candidate_id": None,
                        "source_url": None,
                    })
                elif feedback:
                    acts.append({
                        "name": "Feedback Adjustment",
                        "description": f"Feedback addressed: {feedback}",
                        "category": "leisure",
                        "start_time": "08:00 PM",
                        "estimated_duration": "1 hour",
                        "estimated_cost": 0.0,
                        "candidate_id": None,
                        "source_url": None,
                    })

                days.append({
                    "day": d_idx + 1,
                    "date": curr_date.isoformat(),
                    "focus": interest,
                    "theme": interest,
                    "activities": acts,
                    "estimated_cost": day_budget,
                })

            summary = f"{n_days}-day itinerary in {dest} tailored for {', '.join(interests)}."
            if revision_count > 0:
                summary += f" (Revision {revision_count})"

            return {
                "destination": dest,
                "total_days": n_days,
                "currency": currency,
                "total_estimated_cost": round(day_budget * n_days, 2),
                "days": days,
                "summary": summary,
                "budget_breakdown": {
                    "accommodation": round(target_budget * 0.35, 2),
                    "food": round(target_budget * 0.25, 2),
                    "activities": round(target_budget * 0.20, 2),
                    "local_transportation": round(target_budget * 0.10, 2),
                    "contingency": round(target_budget * 0.10, 2),
                },
                "assumptions": ["Pace is moderate"],
                "warnings": [],
            }

        monkeypatch.setattr("app.graph.nodes.run_planner_agent", fake_planner)
        monkeypatch.setattr("app.agents.planner.run_planner_agent", fake_planner)
        monkeypatch.setattr("app.agents.planner.run_planner_placeholder", fake_planner)


@pytest.fixture(autouse=True)
def mock_validator_groq_in_workflow_tests(request, monkeypatch):
    """Ensure validator does not make external Groq calls in API and workflow tests.

    Allows deterministic validation checks to run fully while keeping test execution
    fast, deterministic, and completely offline without credit consumption.
    """
    filename = request.node.fspath.basename if hasattr(request.node, "fspath") else ""
    if "test_validator" not in filename:
        monkeypatch.setattr(
            "app.agents.itinerary_validator._run_groq_quality_review",
            lambda *args, **kwargs: [],
        )
