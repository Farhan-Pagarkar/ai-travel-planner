"""Pydantic schemas for structured travel itineraries, daily schedules, and activities."""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class ScheduledActivity(BaseModel):
    """An individual planned activity, attraction visit, meal, or leisure block."""

    name: str = Field(description="Activity name or descriptive title.")
    description: str = Field(description="Details on what to see or do.")
    category: str = Field(
        default="sightseeing",
        description="Category: e.g. attraction, event, dining, leisure, transit.",
    )
    start_time: Optional[str] = Field(
        default=None,
        description="Approximate time block or start time (e.g., '09:00 AM' or 'Morning').",
    )
    estimated_duration: Optional[str] = Field(
        default=None,
        description="Estimated duration (e.g., '2 hours', 'Half day').",
    )
    estimated_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated activity cost (non-negative).",
    )
    candidate_id: Optional[str] = Field(
        default=None,
        description="ID of matched candidate from Activity Lookup Tool if a verified attraction/event.",
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Verified source URL matching candidate record, or None for generic blocks.",
    )

    @field_validator("estimated_cost")
    @classmethod
    def validate_non_negative_cost(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Activity estimated cost cannot be negative.")
        return round(float(v), 2)


class ItineraryDay(BaseModel):
    """A single scheduled day within the itinerary."""

    day: int = Field(ge=1, description="Day number (1-indexed).")
    date: str = Field(description="Date in ISO format (YYYY-MM-DD).")
    focus: str = Field(description="Primary theme or focus for the day.")
    theme: Optional[str] = Field(default=None, description="Optional secondary theme label.")
    activities: list[ScheduledActivity] = Field(
        default_factory=list,
        description="List of activities scheduled for this day.",
    )
    estimated_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Total estimated expenditure for this day.",
    )
    cost_breakdown: Optional[dict[str, float]] = Field(
        default=None,
        description="Optional structured daily cost breakdown for accommodation, food, transportation, and activities.",
    )

    @field_validator("estimated_cost")
    @classmethod
    def validate_daily_cost(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Daily estimated cost cannot be negative.")
        return round(float(v), 2)


class Itinerary(BaseModel):
    """Complete structured day-by-day itinerary synthesized by the Planner Agent."""

    destination: str
    total_days: int = Field(ge=1)
    currency: str = "USD"
    total_estimated_cost: float = Field(ge=0.0)
    days: list[ItineraryDay] = Field(default_factory=list)
    summary: str
    budget_breakdown: Optional[dict[str, Any]] = Field(
        default=None,
        description="Category-level allocations from the Budget Calculator Tool.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Planning assumptions (e.g. pace, transit, estimated pricing).",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Caveats, seasonal alerts, or research limitations.",
    )

    @field_validator("total_estimated_cost")
    @classmethod
    def validate_total_cost(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Total estimated cost cannot be negative.")
        return round(float(v), 2)
