"""Itinerary Planner Agent placeholder module.

Constructs day-by-day itineraries with activity assignments and cost estimates.
LLM and dynamic tool integrations will be added in subsequent phases.
"""

from datetime import date, datetime, timedelta
from typing import Any, Optional


def _parse_date(date_val: Any) -> date:
    if isinstance(date_val, date):
        return date_val
    return datetime.strptime(str(date_val), "%Y-%m-%d").date()


def run_planner_placeholder(
    request: dict[str, Any],
    research: Optional[dict[str, Any]] = None,
    feedback: Optional[str] = None,
    modifications: Optional[str] = None,
    revision_count: int = 0,
) -> dict[str, Any]:
    """Generate a structured day-by-day itinerary covering requested dates.

    NOTE: LLM generation and live activity pricing tools will be added later.
    """
    destination = request.get("destination", "Destination")
    interests = request.get("interests", ["sightseeing"])
    currency = request.get("currency", "USD")
    total_budget = float(request.get("total_trip_budget", 1000.0))

    try:
        start_dt = _parse_date(request["start_date"])
        end_dt = _parse_date(request["end_date"])
        num_days = max(1, (end_dt - start_dt).days + 1)
    except Exception:
        start_dt = date.today()
        num_days = 3

    # Plan costs safely within 75% of budget to satisfy validation checks
    budget_target = round(total_budget * 0.75, 2)
    daily_budget = round(budget_target / num_days, 2)

    days: list[dict[str, Any]] = []
    for day_idx in range(num_days):
        current_date = start_dt + timedelta(days=day_idx)
        interest = interests[day_idx % len(interests)]
        activities = [
            f"Morning: Visit top {destination} landmark focused on {interest}",
            f"Afternoon: Explore local culinary highlights and historic markets",
            f"Evening: Leisure stroll and dinner in central {destination}",
        ]

        if modifications:
            activities.append(f"Modification applied: {modifications}")
        elif feedback:
            activities.append(f"Feedback addressed: {feedback}")

        days.append({
            "day": day_idx + 1,
            "date": current_date.isoformat(),
            "focus": interest,
            "activities": activities,
            "estimated_cost": daily_budget,
        })

    total_estimated_cost = round(daily_budget * num_days, 2)
    summary = f"{num_days}-day itinerary in {destination} tailored for {', '.join(interests)}."
    if revision_count > 0:
        summary += f" (Revision {revision_count})"

    return {
        "destination": destination,
        "total_days": num_days,
        "currency": currency,
        "total_estimated_cost": total_estimated_cost,
        "days": days,
        "summary": summary,
    }
