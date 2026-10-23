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
