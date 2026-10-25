from typing import Optional
from pydantic import BaseModel, Field


class DestinationCoordinates(BaseModel):
    """Geographic resolution for a destination."""

    latitude: float
    longitude: float
    resolved_name: str
    timezone: str = "UTC"


class AttractionInfo(BaseModel):
    """Attraction or point of interest."""

    name: str
    description: str
    visitor_tips: Optional[str] = None
    source_url: Optional[str] = None


class LocalReportItem(BaseModel):
    """News item, advisory, historical report, or scheduled event."""

    title: str
    summary: str
    source_url: Optional[str] = None
    publication_date: Optional[str] = Field(
        default=None, description="Actual publication date from source or provider."
    )
    event_date: Optional[str] = Field(
        default=None, description="Explicit event date if confirmed by source."
    )
    category: str = Field(
        default="general",
        description="Category: e.g. transport, closure, festival, weather, safety.",
    )
    potential_travel_impact: str = Field(
        default="low",
        description="Impact level: low, moderate, high, or descriptive notes.",
    )
    is_verified: bool = Field(
        default=True,
        description="False if unconfirmed by a primary or trusted source.",
    )


class DailyWeatherForecast(BaseModel):
    """Daily forecast data within the available forecast window."""

    date: str
    temp_max_c: float
    temp_min_c: float
    precipitation_sum_mm: float
    precipitation_probability_pct: Optional[float] = None
    weather_condition: str


class HistoricalWeatherComparison(BaseModel):
    """Historical weather metrics across equivalent calendar dates from prior years."""

    comparison_years: list[int] = Field(
        default_factory=list, description="Prior calendar years compared (e.g., [2025, 2024, 2023])."
    )
    equivalent_dates_last_year: Optional[str] = None
    typical_temp_max_c: Optional[float] = None
    typical_temp_min_c: Optional[float] = None
    typical_precipitation_mm: Optional[float] = None
    rainy_days_count: Optional[int] = None
    summary: str


class WeatherContext(BaseModel):
    """Structured weather findings distinguishing forecast vs historical data."""

    weather_type: str = Field(
        description="'forecast', 'historical_comparison', 'hybrid', or 'unavailable'"
    )
    forecast_days: list[DailyWeatherForecast] = Field(default_factory=list)
    historical_comparison: Optional[HistoricalWeatherComparison] = None
    notes: Optional[str] = None


class ResearchResult(BaseModel):
    """Complete synthesized research findings produced by the Research Agent."""

    destination: str
    coordinates: Optional[DestinationCoordinates] = None
    attractions: list[AttractionInfo] = Field(default_factory=list)
    local_tips: list[str] = Field(default_factory=list)
    safety_considerations: list[str] = Field(default_factory=list)
    seasonal_considerations: list[str] = Field(default_factory=list)
    recent_local_news: list[LocalReportItem] = Field(
        default_factory=list,
        description="Recent notices, transport updates, and local developments.",
    )
    historical_local_reports: list[LocalReportItem] = Field(
        default_factory=list,
        description="Reports from the corresponding calendar period in prior years.",
    )
    upcoming_local_events: list[LocalReportItem] = Field(
        default_factory=list,
        description="Confirmed or scheduled events during the requested travel dates.",
    )
    weather: Optional[WeatherContext] = None
    source_urls: list[str] = Field(default_factory=list)
    warnings_and_uncertainties: list[str] = Field(default_factory=list)
