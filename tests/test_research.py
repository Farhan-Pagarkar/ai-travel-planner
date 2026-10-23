"""Focused unit and integration tests for web search, weather, and Research Agent."""

from datetime import date, timedelta
from typing import Any
from unittest.mock import MagicMock
import httpx
import pytest

from app.agents.researcher import run_researcher_agent
from app.schemas.research import ResearchResult
from app.tools.weather import WeatherClient, gather_weather_intelligence
from app.tools.web_search import SearchResultItem, SerperClient, gather_destination_research


# ---------------------------------------------------------------------------
# 1. Serper Web & News Search Tests
# ---------------------------------------------------------------------------


def test_serper_search_parsing(monkeypatch):
    """Test parsing standard organic search results from Serper."""
    mock_payload = {
        "organic": [
            {
                "title": "Kyoto Travel Guide",
                "link": "https://example.com/kyoto-guide",
                "snippet": "Discover the best historic temples in Kyoto.",
                "date": "2026-03-10",
                "position": 1,
            }
        ]
    }

    def mock_post(self, url, *args, **kwargs):
        req = httpx.Request("POST", url)
        return httpx.Response(status_code=200, json=mock_payload, request=req)

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    client = SerperClient(api_key="fake-test-key")
    results = client.search("Kyoto attractions", search_type="search")

    assert len(results) == 1
    item = results[0]
    assert item.title == "Kyoto Travel Guide"
    assert item.url == "https://example.com/kyoto-guide"
    assert "historic temples" in item.snippet
    assert item.publication_date == "2026-03-10"


def test_serper_news_parsing(monkeypatch):
    """Test parsing news results with publication dates and sources."""
    mock_news_payload = {
        "news": [
            {
                "title": "Kyoto Railway Announces Schedule Revision",
                "link": "https://news.example.com/kyoto-rail",
                "snippet": "City transport updates commuter routes for autumn.",
                "date": "3 days ago",
                "source": "Japan Daily News",
            }
        ]
    }

    def mock_post(self, url, *args, **kwargs):
        req = httpx.Request("POST", url)
        return httpx.Response(status_code=200, json=mock_news_payload, request=req)

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    client = SerperClient(api_key="fake-test-key")
    results = client.search("Kyoto transport news", search_type="news")

    assert len(results) == 1
    item = results[0]
    assert item.title == "Kyoto Railway Announces Schedule Revision"
    assert item.url == "https://news.example.com/kyoto-rail"
    assert item.source == "Japan Daily News"
    assert item.publication_date == "3 days ago"


def test_serper_missing_api_key(monkeypatch):
    """Test missing Serper API key raises clear error without exposing credentials."""
    monkeypatch.setenv("SERPER_API_KEY", "")
    client = SerperClient(api_key="")

    with pytest.raises(ValueError) as excinfo:
        client.search("test query")
    assert "SERPER_API_KEY is not configured" in str(excinfo.value)


def test_serper_http_errors_and_rate_limits(monkeypatch):
    """Test HTTP status errors: 401 auth failure and 429 rate limit."""
    # 401 Auth Failure
    def mock_401(self, url, *args, **kwargs):
        req = httpx.Request("POST", url)
        return httpx.Response(status_code=401, json={"message": "Unauthorized"}, request=req)

    monkeypatch.setattr(httpx.Client, "post", mock_401)
    client = SerperClient(api_key="bad-key")
    with pytest.raises(PermissionError) as exc_auth:
        client.search("query")
    assert "authentication failed" in str(exc_auth.value).lower()

    # 429 Rate Limit
    def mock_429(self, url, *args, **kwargs):
        req = httpx.Request("POST", url)
        return httpx.Response(status_code=429, json={"message": "Too Many Requests"}, request=req)

    monkeypatch.setattr(httpx.Client, "post", mock_429)
    with pytest.raises(RuntimeError) as exc_rate:
        client.search("query")
    assert "rate limit" in str(exc_rate.value).lower()


def test_serper_timeout(monkeypatch):
    """Test client timeout handling."""
    def mock_timeout(self, url, *args, **kwargs):
        raise httpx.TimeoutException("Connection timed out")

    monkeypatch.setattr(httpx.Client, "post", mock_timeout)
    client = SerperClient(api_key="valid-key")
    with pytest.raises(TimeoutError):
        client.search("query")


# ---------------------------------------------------------------------------
# 2. Open-Meteo Weather Tool Tests
# ---------------------------------------------------------------------------


def test_weather_geocoding_success(monkeypatch):
    """Test successful destination geocoding."""
    mock_geo_resp = {
        "results": [
            {
                "name": "Kyoto",
                "country": "Japan",
                "latitude": 35.02107,
                "longitude": 135.75385,
                "timezone": "Asia/Tokyo",
            }
        ]
    }

    def mock_get(self, url, *args, **kwargs):
        req = httpx.Request("GET", url)
        return httpx.Response(status_code=200, json=mock_geo_resp, request=req)

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    client = WeatherClient()
    coords = client.geocode_destination("Kyoto")

    assert coords is not None
    assert coords.latitude == 35.02107
    assert coords.longitude == 135.75385
    assert coords.resolved_name == "Kyoto, Japan"
    assert coords.timezone == "Asia/Tokyo"


def test_weather_geocoding_unknown_location(monkeypatch):
    """Test geocoding an unknown location safely returns None."""
    def mock_empty_geo(self, url, *args, **kwargs):
        req = httpx.Request("GET", url)
        return httpx.Response(status_code=200, json={"results": []}, request=req)

    monkeypatch.setattr(httpx.Client, "get", mock_empty_geo)

    client = WeatherClient()
    coords = client.geocode_destination("NonExistentCityXYZ123")
    assert coords is None


def test_weather_forecast_within_horizon(monkeypatch):
    """Test retrieving forecast data for dates within the 16-day forecast horizon."""
    today = date(2026, 10, 1)
    trip_start = "2026-10-02"
    trip_end = "2026-10-04"

    mock_geo = {
        "results": [{"name": "Rome", "country": "Italy", "latitude": 41.9, "longitude": 12.5, "timezone": "Europe/Rome"}]
    }
    mock_forecast = {
        "daily": {
            "time": ["2026-10-02", "2026-10-03", "2026-10-04"],
            "temperature_2m_max": [22.5, 23.0, 21.8],
            "temperature_2m_min": [13.2, 14.1, 12.9],
            "precipitation_sum": [0.0, 2.5, 0.0],
            "precipitation_probability_max": [10, 60, 15],
            "weather_code": [1, 61, 0],
        }
    }

    def mock_get(self, url, params=None, *args, **kwargs):
        req = httpx.Request("GET", url)
        if "geocoding" in url:
            return httpx.Response(status_code=200, json=mock_geo, request=req)
        if "forecast" in url:
            return httpx.Response(status_code=200, json=mock_forecast, request=req)
        return httpx.Response(status_code=404, request=req)

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    coords, weather = gather_weather_intelligence("Rome", trip_start, trip_end, reference_today=today)

    assert coords is not None
    assert weather.weather_type == "forecast"
    assert len(weather.forecast_days) == 3
    assert weather.forecast_days[0].date == "2026-10-02"
    assert weather.forecast_days[0].temp_max_c == 22.5
    assert weather.forecast_days[1].weather_condition == "Slight rain"
    assert weather.forecast_days[1].precipitation_probability_pct == 60.0


def test_weather_historical_comparison_outside_horizon(monkeypatch):
    """Test historical climate comparison across prior years for dates outside forecast horizon."""
    today = date(2026, 5, 1)
    trip_start = "2026-11-01"
    trip_end = "2026-11-04"

    mock_geo = {
        "results": [{"name": "Kyoto", "country": "Japan", "latitude": 35.0, "longitude": 135.7, "timezone": "Asia/Tokyo"}]
    }
    mock_archive = {
        "daily": {
            "time": ["2025-11-01", "2025-11-02", "2025-11-03", "2025-11-04"],
            "temperature_2m_max": [19.0, 20.0, 18.5, 19.5],
            "temperature_2m_min": [10.0, 11.0, 9.5, 10.5],
            "precipitation_sum": [0.0, 5.0, 0.0, 0.0],
            "weather_code": [1, 61, 0, 1],
        }
    }

    def mock_get(self, url, params=None, *args, **kwargs):
        req = httpx.Request("GET", url)
        if "geocoding" in url:
            return httpx.Response(status_code=200, json=mock_geo, request=req)
        if "archive" in url:
            return httpx.Response(status_code=200, json=mock_archive, request=req)
        return httpx.Response(status_code=404, request=req)

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    coords, weather = gather_weather_intelligence("Kyoto", trip_start, trip_end, reference_today=today)

    assert coords is not None
    assert weather.weather_type == "historical_comparison"
    assert weather.historical_comparison is not None
    assert 2025 in weather.historical_comparison.comparison_years
    assert weather.historical_comparison.typical_temp_max_c is not None
    assert "historical" in weather.historical_comparison.summary.lower()


def test_weather_hybrid_overlap(monkeypatch):
    """Test hybrid weather behavior when dates partly overlap forecast horizon."""
    today = date(2026, 10, 1)
    # Forecast horizon is 2026-10-01 to 2026-10-16.
    # Trip is 2026-10-15 to 2026-10-18 (partly in, partly out).
    trip_start = "2026-10-15"
    trip_end = "2026-10-18"

    mock_geo = {
        "results": [{"name": "Tokyo", "latitude": 35.6, "longitude": 139.6, "timezone": "Asia/Tokyo"}]
    }
    mock_forecast = {
        "daily": {
            "time": ["2026-10-15", "2026-10-16"],
            "temperature_2m_max": [21.0, 20.5],
            "temperature_2m_min": [15.0, 14.5],
            "precipitation_sum": [0.0, 1.0],
            "precipitation_probability_max": [20, 40],
            "weather_code": [1, 2],
        }
    }
    mock_archive = {
        "daily": {
            "temperature_2m_max": [22.0, 21.0],
            "temperature_2m_min": [14.0, 13.0],
            "precipitation_sum": [0.0, 0.0],
        }
    }

    def mock_get(self, url, *args, **kwargs):
        req = httpx.Request("GET", url)
        if "geocoding" in url:
            return httpx.Response(status_code=200, json=mock_geo, request=req)
        if "forecast" in url:
            return httpx.Response(status_code=200, json=mock_forecast, request=req)
        if "archive" in url:
            return httpx.Response(status_code=200, json=mock_archive, request=req)
        return httpx.Response(status_code=404, request=req)

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    coords, weather = gather_weather_intelligence("Tokyo", trip_start, trip_end, reference_today=today)

    assert weather.weather_type == "hybrid"
    assert len(weather.forecast_days) == 2
    assert weather.historical_comparison is not None
    assert "partially overlaps" in weather.notes.lower()


def test_weather_missing_data_and_failures(monkeypatch):
    """Test graceful handling when weather APIs fail or destination is unresolved."""
    def mock_fail(self, url, *args, **kwargs):
        raise httpx.RequestError("Network unreachable")

    monkeypatch.setattr(httpx.Client, "get", mock_fail)

    coords, weather = gather_weather_intelligence("Unknown City", "2026-11-01", "2026-11-04")
    assert coords is None
    assert weather.weather_type == "unavailable"
    assert "unable to resolve" in weather.notes.lower()


# ---------------------------------------------------------------------------
# 3. Groq Research Synthesis & Agent Tests
# ---------------------------------------------------------------------------


def test_groq_synthesis_and_validation():
    """Test Research Agent synthesis using mock Serper, Weather, and Groq clients."""
    mock_serper = MagicMock(spec=SerperClient)
    mock_serper.search.side_effect = lambda q, search_type="search", limit=5: [
        SearchResultItem(
            title="Kyoto Guide",
            url="https://example.com/kyoto",
            snippet="Visit Kinkaku-ji and Arashiyama.",
            source="Travel Weekly",
            publication_date="2026-01-15",
        )
    ]

    mock_weather = MagicMock(spec=WeatherClient)
    mock_weather.geocode_destination.return_value = None

    mock_groq_resp = {
        "destination": "Kyoto, Japan",
        "attractions": [
            {
                "name": "Kinkaku-ji",
                "description": "Golden Pavilion Zen temple.",
                "visitor_tips": "Arrive at opening.",
                "source_url": "https://example.com/kyoto",
            }
        ],
        "local_tips": ["Buy a Kyoto bus day pass."],
        "safety_considerations": ["Keep hydration handy during summer."],
        "seasonal_considerations": ["Pleasant autumn colors."],
        "recent_local_news": [
            {
                "title": "Subway Line Expands Service",
                "summary": "Kyoto subway announced extended evening hours.",
                "source_url": "https://example.com/subway",
                "publication_date": "2026-10-10",
                "event_date": None,
                "category": "transport",
                "potential_travel_impact": "low",
                "is_verified": True,
            }
        ],
        "historical_local_reports": [
            {
                "title": "Strong Fall Crowds in 2025",
                "summary": "Historic areas saw high visitor density.",
                "source_url": "https://example.com/crowds-2025",
                "publication_date": "2025-11-20",
                "event_date": None,
                "category": "travel",
                "potential_travel_impact": "moderate",
                "is_verified": True,
            }
        ],
        "upcoming_local_events": [],
        "source_urls": ["https://example.com/kyoto"],
        "warnings_and_uncertainties": ["Check temple entry times in advance."],
    }

    mock_groq = MagicMock()
    mock_choice = MagicMock()
    import json
    mock_choice.message.content = json.dumps(mock_groq_resp)
    mock_groq.chat.completions.create.return_value.choices = [mock_choice]

    request = {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-04",
        "interests": ["temples", "gardens"],
    }

    result = run_researcher_agent(
        request=request,
        serper_client=mock_serper,
        weather_client=mock_weather,
        groq_client=mock_groq,
        groq_model="openai/gpt-oss-20b",
    )

    assert result["destination"] == "Kyoto, Japan"
    assert len(result["attractions"]) == 1
    assert result["attractions"][0]["name"] == "Kinkaku-ji"
    assert len(result["recent_local_news"]) == 1
    assert len(result["historical_local_reports"]) == 1
    # Check that source URL from Serper was preserved
    assert "https://example.com/kyoto" in result["source_urls"]


def test_groq_missing_api_key(monkeypatch):
    """Test missing Groq API key raises clear descriptive error."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    mock_serper = MagicMock(spec=SerperClient)

    request = {
        "destination": "Kyoto",
        "start_date": "2026-11-01",
        "end_date": "2026-11-04",
    }

    with pytest.raises(ValueError) as excinfo:
        run_researcher_agent(request, serper_client=mock_serper, groq_client=None)
    assert "GROQ_API_KEY is not configured" in str(excinfo.value)


def test_groq_invalid_output_handling():
    """Test handling when Groq returns invalid JSON."""
    mock_serper = MagicMock(spec=SerperClient)
    mock_serper.search.return_value = []

    mock_weather = MagicMock(spec=WeatherClient)
    mock_weather.geocode_destination.return_value = None

    mock_groq = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Not a JSON document."
    mock_groq.chat.completions.create.return_value.choices = [mock_choice]

    request = {
        "destination": "Kyoto",
        "start_date": "2026-11-01",
        "end_date": "2026-11-04",
    }

    with pytest.raises(RuntimeError) as excinfo:
        run_researcher_agent(
            request,
            serper_client=mock_serper,
            weather_client=mock_weather,
            groq_client=mock_groq,
        )
    assert "invalid json" in str(excinfo.value).lower()


def test_research_agent_source_provenance_and_separation():
    """Verify source provenance is retained and news vs historical reports are distinguished."""
    mock_serper = MagicMock(spec=SerperClient)
    mock_serper.search.side_effect = lambda q, search_type="search", limit=5: [
        SearchResultItem(
            title="Discovery Article",
            url="https://source.example.com/article-1",
            snippet="Information about Kyoto culture.",
            source="Travel Press",
            publication_date="2026-05-01",
        )
    ]

    mock_weather = MagicMock(spec=WeatherClient)
    mock_weather.geocode_destination.return_value = None

    import json
    mock_payload = {
        "destination": "Kyoto",
        "attractions": [],
        "local_tips": [],
        "safety_considerations": [],
        "seasonal_considerations": [],
        "recent_local_news": [
            {
                "title": "Recent news item",
                "summary": "Recent happening.",
                "source_url": "https://source.example.com/article-1",
                "publication_date": "2026-05-01",
                "event_date": None,
                "category": "transport",
                "potential_travel_impact": "low",
                "is_verified": True,
            }
        ],
        "historical_local_reports": [
            {
                "title": "Historical 2025 event",
                "summary": "Event from prior year.",
                "source_url": "https://source.example.com/article-historical",
                "publication_date": "2025-05-01",
                "event_date": None,
                "category": "weather",
                "potential_travel_impact": "low",
                "is_verified": True,
            }
        ],
        "upcoming_local_events": [],
        "source_urls": ["https://source.example.com/article-1"],
        "warnings_and_uncertainties": [],
    }

    mock_groq = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_payload)
    mock_groq.chat.completions.create.return_value.choices = [mock_choice]

    result = run_researcher_agent(
        request={"destination": "Kyoto", "start_date": "2026-11-01", "end_date": "2026-11-04"},
        serper_client=mock_serper,
        weather_client=mock_weather,
        groq_client=mock_groq,
    )

    assert "https://source.example.com/article-1" in result["source_urls"]
    assert len(result["recent_local_news"]) == 1
    assert result["recent_local_news"][0]["publication_date"] == "2026-05-01"
    assert len(result["historical_local_reports"]) == 1
    assert result["historical_local_reports"][0]["publication_date"] == "2025-05-01"


# ---------------------------------------------------------------------------
# 4. Live Smoke Test (runs only if credentials are configured)
# ---------------------------------------------------------------------------


def test_live_research_smoke_test():
    """Optional live smoke test running real Serper, Open-Meteo, and Groq requests.

    Skipped if valid SERPER_API_KEY and GROQ_API_KEY are not present in environment.
    """
    import os
    serper_key = os.getenv("SERPER_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if not serper_key or not groq_key or "your_" in serper_key or "your_" in groq_key:
        pytest.skip("Live credentials for Serper or Groq not available; skipping live smoke test.")

    request = {
        "destination": "Kyoto, Japan",
        "start_date": "2026-11-01",
        "end_date": "2026-11-03",
        "interests": ["temples", "culinary"],
    }

    result = run_researcher_agent(request)
    assert result["destination"] == "Kyoto, Japan"
    assert "attractions" in result
    assert "weather" in result
    assert "source_urls" in result
    assert len(result["source_urls"]) > 0
