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
    mock_serper.search.return_value = [
        SearchResultItem(title="Kyoto Guide", url="https://example.com/kyoto", snippet="Guide info")
    ]

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


@pytest.mark.live
def test_live_research_smoke_test(request):
    """Live integration smoke test running real Serper, Open-Meteo, and Groq requests.

    Explicit opt-in only: runs only when requested via RUN_LIVE_TESTS=1 or `pytest -m live`.
    Skipped by default during standard test runs to avoid consuming external API credits.
    """
    import os
    run_live = (
        os.getenv("RUN_LIVE_TESTS", "").lower() in ("1", "true", "yes")
        or request.config.getoption("-m") == "live"
        or "live" in getattr(request.config.option, "markexpr", "")
    )
    if not run_live:
        pytest.skip("Live smoke test skipped by default. Run with `RUN_LIVE_TESTS=1 pytest` or `pytest -m live` to opt in.")

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


# ---------------------------------------------------------------------------
# 5. Weather Edge Cases: Cross-Year, Leap-Day, and Hybrid Window Range Tests
# ---------------------------------------------------------------------------


def test_weather_historical_cross_year_dates(monkeypatch):
    """Test historical date shifting across calendar year boundaries (e.g. Dec 30 to Jan 2)."""
    captured_params = []

    def mock_get(self, url, params=None, *args, **kwargs):
        req = httpx.Request("GET", url)
        if "archive" in url and params:
            captured_params.append(dict(params))
            mock_data = {
                "daily": {
                    "temperature_2m_max": [10.0, 9.5, 11.0, 10.5],
                    "temperature_2m_min": [2.0, 1.5, 3.0, 2.5],
                    "precipitation_sum": [0.0, 1.0, 0.0, 0.0],
                }
            }
            return httpx.Response(status_code=200, json=mock_data, request=req)
        return httpx.Response(status_code=404, request=req)

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    client = WeatherClient()
    start_dt = date(2026, 12, 30)
    end_dt = date(2027, 1, 2)

    hist_comp = client.get_historical_comparison(
        lat=35.0, lon=135.7, tz="Asia/Tokyo", start_date=start_dt, end_date=end_dt, years_back=2
    )

    assert hist_comp is not None
    assert len(captured_params) == 2
    # Offset 1 (prior year): Dec 30, 2025 to Jan 2, 2026
    assert captured_params[0]["start_date"] == "2025-12-30"
    assert captured_params[0]["end_date"] == "2026-01-02"
    # Offset 2: Dec 30, 2024 to Jan 2, 2025
    assert captured_params[1]["start_date"] == "2024-12-30"
    assert captured_params[1]["end_date"] == "2025-01-02"
    # Verify equivalent dates string preserves cross-year boundary
    assert hist_comp.equivalent_dates_last_year == "2025-12-30 to 2026-01-02"
    assert "2025-2026" in hist_comp.summary


def test_weather_historical_leap_day_dates(monkeypatch):
    """Test historical date shifting when start or end date is February 29 shifted to non-leap years."""
    captured_params = []

    def mock_get(self, url, params=None, *args, **kwargs):
        req = httpx.Request("GET", url)
        if "archive" in url and params:
            captured_params.append(dict(params))
            mock_data = {
                "daily": {
                    "temperature_2m_max": [12.0, 13.0],
                    "temperature_2m_min": [4.0, 5.0],
                    "precipitation_sum": [0.0, 0.5],
                }
            }
            return httpx.Response(status_code=200, json=mock_data, request=req)
        return httpx.Response(status_code=404, request=req)

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    client = WeatherClient()
    # 2028 is a leap year; trip includes Feb 29
    start_dt = date(2028, 2, 28)
    end_dt = date(2028, 3, 1)

    hist_comp = client.get_historical_comparison(
        lat=48.8, lon=2.3, tz="Europe/Paris", start_date=start_dt, end_date=end_dt, years_back=1
    )

    assert hist_comp is not None
    assert len(captured_params) == 1
    # 2027 is not a leap year; Feb 28 remains Feb 28, March 1 remains March 1
    assert captured_params[0]["start_date"] == "2027-02-28"
    assert captured_params[0]["end_date"] == "2027-03-01"

    # Now test starting directly on leap day (Feb 29, 2028)
    captured_params.clear()
    hist_comp_feb29 = client.get_historical_comparison(
        lat=48.8, lon=2.3, tz="Europe/Paris", start_date=date(2028, 2, 29), end_date=date(2028, 3, 2), years_back=1
    )
    assert hist_comp_feb29 is not None
    assert captured_params[0]["start_date"] == "2027-02-28"
    assert captured_params[0]["end_date"] == "2027-03-02"


def test_weather_hybrid_window_ranges():
    """Verify that hybrid weather windows query historical comparisons ONLY for dates beyond the 16-day horizon."""
    mock_client = MagicMock(spec=WeatherClient)
    from app.schemas.research import DestinationCoordinates, HistoricalWeatherComparison
    mock_coords = DestinationCoordinates(latitude=35.6, longitude=139.6, resolved_name="Tokyo, Japan", timezone="Asia/Tokyo")
    mock_client.geocode_destination.return_value = mock_coords
    mock_client.get_forecast.return_value = []
    mock_client.get_historical_comparison.return_value = HistoricalWeatherComparison(
        comparison_years=[2025, 2024],
        summary="Historical summary for Tokyo.",
    )

    # Reference today: 2026-10-01. Forecast horizon is 2026-10-01 to 2026-10-16.
    # Trip: 2026-10-14 to 2026-10-19.
    # Covered by forecast: 2026-10-14, 2026-10-15, 2026-10-16.
    # Beyond forecast horizon: 2026-10-17 to 2026-10-19.
    coords, weather = gather_weather_intelligence(
        destination="Tokyo",
        start_date_str="2026-10-14",
        end_date_str="2026-10-19",
        client=mock_client,
        reference_today=date(2026, 10, 1),
    )

    assert weather.weather_type == "hybrid"
    # Verify get_historical_comparison was called with ONLY dates beyond the horizon (2026-10-17 to 2026-10-19)
    mock_client.get_historical_comparison.assert_called_once_with(
        35.6, 139.6, "Asia/Tokyo", date(2026, 10, 17), date(2026, 10, 19)
    )
    # Verify notes accurately describe both forecast and historical ranges
    assert "2026-10-14 to 2026-10-16" in weather.notes
    assert "2026-10-17 to 2026-10-19" in weather.notes


# ---------------------------------------------------------------------------
# 6. Structured Weather Context & LLM Evidence Grounding Tests
# ---------------------------------------------------------------------------


def test_weather_data_present_in_llm_evidence():
    """Verify that detailed forecast observations and historical comparison figures are in LLM prompt evidence."""
    from app.schemas.research import DailyWeatherForecast, HistoricalWeatherComparison, WeatherContext
    from app.agents.researcher import _format_evidence_package

    forecast = [
        DailyWeatherForecast(
            date="2026-10-15",
            temp_max_c=22.4,
            temp_min_c=14.1,
            precipitation_sum_mm=3.2,
            precipitation_probability_pct=75.0,
            weather_condition="Moderate rain",
        )
    ]
    historical = HistoricalWeatherComparison(
        comparison_years=[2025, 2024],
        equivalent_dates_last_year="2025-10-15 to 2025-10-18",
        typical_temp_max_c=21.0,
        typical_temp_min_c=13.5,
        typical_precipitation_mm=5.5,
        rainy_days_count=1,
        summary="Typical October conditions with mild temperatures.",
    )
    weather_ctx = WeatherContext(
        weather_type="hybrid",
        forecast_days=forecast,
        historical_comparison=historical,
        notes="Hybrid weather context.",
    )

    search_evidence = {
        "destination_info": [SearchResultItem(title="Tokyo Tower", url="https://example.com/tokyo", snippet="Landmark")]
    }

    evidence_text = _format_evidence_package(
        destination="Tokyo",
        start_date="2026-10-15",
        end_date="2026-10-18",
        interests=["culture"],
        search_data=search_evidence,
        weather_context=weather_ctx,
    )

    # Check that daily forecast values are present
    assert "22.4°C" in evidence_text
    assert "14.1°C" in evidence_text
    assert "3.2mm" in evidence_text
    assert "Moderate rain" in evidence_text
    assert "75.0%" in evidence_text
    # Check that historical comparison figures are present
    assert "21.0°C" in evidence_text
    assert "13.5°C" in evidence_text
    assert "5.5mm" in evidence_text
    assert "[2025, 2024]" in evidence_text
    assert "Typical October conditions" in evidence_text


# ---------------------------------------------------------------------------
# 7. Source Provenance & URL Grounding Tests
# ---------------------------------------------------------------------------


def test_source_provenance_validation_removes_fabricated_urls():
    """Verify source provenance validation removes fabricated URLs, retains genuine URLs, and handles missing URLs."""
    import json
    mock_serper = MagicMock(spec=SerperClient)
    # Search tool returns 2 genuine URLs
    genuine_url_1 = "https://example.com/kyoto-temple"
    genuine_url_2 = "https://example.com/kyoto-news"
    mock_serper.search.side_effect = lambda q, search_type="search", limit=5: [
        SearchResultItem(title="Kyoto Guide", url=genuine_url_1, snippet="Kyoto guide info"),
        SearchResultItem(title="Kyoto News", url=genuine_url_2, snippet="Kyoto news info"),
    ]

    mock_weather = MagicMock(spec=WeatherClient)
    mock_weather.geocode_destination.return_value = None

    # Groq returns a mix: genuine URL, fabricated URL, missing URL
    fabricated_url = "https://hallucinated-travel-blog.com/fake-shrine"
    fabricated_news_url = "https://hallucinated-news.com/fake-news"

    mock_llm_json = {
        "destination": "Kyoto",
        "attractions": [
            {
                "name": "Kinkaku-ji",
                "description": "Golden Pavilion",
                "visitor_tips": "Go early",
                "source_url": genuine_url_1,  # Genuine retrieved URL
            },
            {
                "name": "Fabricated Shrine",
                "description": "Non-existent or ungrounded URL",
                "visitor_tips": None,
                "source_url": fabricated_url,  # Fabricated URL
            },
            {
                "name": "Arashiyama Bamboo Grove",
                "description": "Bamboo grove",
                "visitor_tips": None,
                "source_url": None,  # Missing source URL
            },
        ],
        "local_tips": ["Use public transit"],
        "safety_considerations": ["Stay hydrated"],
        "seasonal_considerations": ["Autumn foliage"],
        "recent_local_news": [
            {
                "title": "Kyoto Railway News",
                "summary": "Service expansion",
                "source_url": genuine_url_2,  # Genuine retrieved URL
                "publication_date": "2026-10-01",
                "event_date": None,
                "category": "transport",
                "potential_travel_impact": "low",
                "is_verified": True,
            }
        ],
        "historical_local_reports": [
            {
                "title": "Historical 2025 Report",
                "summary": "Past festival notes",
                "source_url": fabricated_news_url,  # Fabricated URL
                "publication_date": "2025-10-01",
                "event_date": None,
                "category": "culture",
                "potential_travel_impact": "low",
                "is_verified": True,
            }
        ],
        "upcoming_local_events": [
            {
                "title": "Local Market",
                "summary": "Weekend market",
                "source_url": None,  # Missing source URL
                "publication_date": None,
                "event_date": "2026-11-02",
                "category": "shopping",
                "potential_travel_impact": "low",
                "is_verified": True,
            }
        ],
        "source_urls": [genuine_url_1, fabricated_url],
        "warnings_and_uncertainties": [],
    }

    mock_groq = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_llm_json)
    mock_groq.chat.completions.create.return_value.choices = [mock_choice]

    result = run_researcher_agent(
        request={"destination": "Kyoto", "start_date": "2026-11-01", "end_date": "2026-11-03"},
        serper_client=mock_serper,
        weather_client=mock_weather,
        groq_client=mock_groq,
    )

    # 1. Genuine URL is preserved
    assert result["attractions"][0]["source_url"] == genuine_url_1
    assert result["recent_local_news"][0]["source_url"] == genuine_url_2

    # 2. Fabricated URL is removed (set to None)
    assert result["attractions"][1]["source_url"] is None
    assert result["historical_local_reports"][0]["source_url"] is None

    # 3. Missing URL remains None without crashing
    assert result["attractions"][2]["source_url"] is None
    assert result["upcoming_local_events"][0]["source_url"] is None

    # 4. Fabricated URL is omitted from top-level source_urls; genuine URLs are retained
    assert fabricated_url not in result["source_urls"]
    assert genuine_url_1 in result["source_urls"]
    assert genuine_url_2 in result["source_urls"]

    # 5. Warnings are recorded for the removed unsupported URLs
    warnings = " ".join(result["warnings_and_uncertainties"])
    assert fabricated_url in warnings
    assert fabricated_news_url in warnings


# ---------------------------------------------------------------------------
# 8. Web Search Failure Handling & Credential Masking Tests
# ---------------------------------------------------------------------------


def test_partial_search_failure_recorded_in_warnings():
    """Test that a partial search failure (one category fails) is recorded in warnings without leaking secrets."""
    import json
    mock_serper = MagicMock(spec=SerperClient)

    # Destination search succeeds, news search fails with sensitive token in error
    def mock_search(query, search_type="search", limit=5):
        if search_type == "news":
            raise PermissionError("401 Unauthorized: Invalid X-API-KEY secret_key_12345abcde")
        return [SearchResultItem(title="Kyoto Guide", url="https://example.com/kyoto", snippet="Guide")]

    mock_serper.search.side_effect = mock_search

    mock_weather = MagicMock(spec=WeatherClient)
    mock_weather.geocode_destination.return_value = None

    mock_llm_json = {
        "destination": "Kyoto",
        "attractions": [
            {"name": "Kyoto Shrine", "description": "Shrine", "visitor_tips": None, "source_url": "https://example.com/kyoto"}
        ],
        "local_tips": [],
        "safety_considerations": [],
        "seasonal_considerations": [],
        "recent_local_news": [],
        "historical_local_reports": [],
        "upcoming_local_events": [],
        "source_urls": ["https://example.com/kyoto"],
        "warnings_and_uncertainties": [],
    }
    mock_groq = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_llm_json)
    mock_groq.chat.completions.create.return_value.choices = [mock_choice]

    result = run_researcher_agent(
        request={"destination": "Kyoto", "start_date": "2026-11-01", "end_date": "2026-11-03"},
        serper_client=mock_serper,
        weather_client=mock_weather,
        groq_client=mock_groq,
    )

    # Check that research succeeded with partial evidence
    assert len(result["attractions"]) == 1
    # Check that category failure is recorded in warnings
    warnings_str = " ".join(result["warnings_and_uncertainties"])
    assert "recent news" in warnings_str
    # Verify no credentials leaked
    assert "secret_key_12345abcde" not in warnings_str
    assert "x-api-key" not in warnings_str.lower()


def test_complete_search_failure_raises_error():
    """Test that a complete search failure raises a clear error and does not fabricate destination results."""
    mock_serper = MagicMock(spec=SerperClient)
    mock_serper.search.side_effect = httpx.ConnectError("Network connection unreachable with token=secret_token_abc")

    mock_weather = MagicMock(spec=WeatherClient)
    mock_weather.geocode_destination.return_value = None
    mock_groq = MagicMock()

    with pytest.raises(RuntimeError) as excinfo:
        run_researcher_agent(
            request={"destination": "Kyoto", "start_date": "2026-11-01", "end_date": "2026-11-03"},
            serper_client=mock_serper,
            weather_client=mock_weather,
            groq_client=mock_groq,
        )

    err_msg = str(excinfo.value)
    assert "Destination research failed" in err_msg
    # Verify credentials/tokens are never leaked
    assert "secret_token_abc" not in err_msg
    # Verify Groq was never called to fabricate results
    mock_groq.chat.completions.create.assert_not_called()



def test_weather_intelligence_handles_past_dates_without_index_error():
    """Verify gather_weather_intelligence handles past dates cleanly without raising IndexError: list index out of range."""
    from app.tools.weather import gather_weather_intelligence, WeatherClient
    from app.schemas.research import HistoricalWeatherComparison

    mock_client = MagicMock(spec=WeatherClient)
    mock_client.geocode_destination.return_value = MagicMock(latitude=35.0, longitude=135.7, timezone="Asia/Tokyo")
    mock_client.get_historical_comparison.return_value = HistoricalWeatherComparison(
        comparison_years=[2025, 2024],
        typical_temp_max_c=18.0,
        typical_temp_min_c=8.0,
        typical_precipitation_mm=5.0,
        rainy_days_count=1,
        summary="Cool conditions.",
    )

    # Calling with past dates relative to reference_today
    coords, weather = gather_weather_intelligence(
        destination="Kyoto",
        start_date_str="2026-04-09",
        end_date_str="2026-04-19",
        client=mock_client,
        reference_today=date(2026, 10, 25),
    )

    assert weather.weather_type == "historical_comparison"
    assert weather.historical_comparison is not None
    assert mock_client.get_historical_comparison.called
    assert "historical past dates" in weather.notes.lower()
