from datetime import date
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class TravelRequest(BaseModel):
    destination: str = Field(..., description="Target destination for the trip.")
    start_date: date = Field(..., description="Trip start date (YYYY-MM-DD).")
    end_date: date = Field(..., description="Trip end date (YYYY-MM-DD).")
    total_trip_budget: float = Field(..., gt=0, description="Total budget, must be positive.")
    currency: str = Field(default="USD", description="Currency code (e.g., USD, EUR).")
    interests: list[str] = Field(..., min_length=1, description="List of traveler interests.")
    traveler_count: int = Field(default=1, gt=0, description="Number of travelers, must be at least 1.")
    preferences: Optional[dict[str, Any]] = Field(default=None, description="Optional traveler preferences.")

    @field_validator("destination", "currency")
    @classmethod
    def check_non_empty_strings(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field must not be empty or whitespace.")
        return value.strip()

    @field_validator("interests")
    @classmethod
    def check_interests_content(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("Interests must contain at least one non-empty string.")
        return cleaned

    @model_validator(mode="after")
    def validate_travel_dates(self) -> "TravelRequest":
        today = date.today()
        if self.start_date < today:
            raise ValueError(
                f"Trip start_date ({self.start_date.isoformat()}) cannot be earlier than the current date ({today.isoformat()})."
            )
        if self.end_date < self.start_date:
            raise ValueError(
                f"Trip end_date ({self.end_date.isoformat()}) cannot precede start_date ({self.start_date.isoformat()})."
            )
        duration_days = (self.end_date - self.start_date).days + 1
        if duration_days > 14:
            raise ValueError(
                f"Trip duration of {duration_days} days exceeds the maximum allowed limit of 14 calendar days inclusive."
            )
        return self
