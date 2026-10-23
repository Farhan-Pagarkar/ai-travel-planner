"""Itinerary Validator Agent placeholder module.

Validates constraints, schedule consistency, and budget boundaries.
Advanced constraint rules and LLM hallucination checks will be added later.
"""

from datetime import date, datetime
from typing import Any, Optional


def _parse_date(date_val: Any) -> date:
    if isinstance(date_val, date):
        return date_val
    return datetime.strptime(str(date_val), "%Y-%m-%d").date()


def run_validator_placeholder(
    request: dict[str, Any],
    draft_itinerary: Optional[dict[str, Any]],
) -> list[str]:
    """Perform deterministic validation on draft itinerary.

    Returns a list of validation issue descriptions, or empty list if valid.
    NOTE: Advanced LLM-based constraint validation will be added in subsequent phases.
    """
    issues: list[str] = []

    if not draft_itinerary:
        return ["No draft itinerary found to validate."]

    # 1. Verify trip duration matches requested dates
    try:
        start_dt = _parse_date(request["start_date"])
        end_dt = _parse_date(request["end_date"])
        expected_days = max(1, (end_dt - start_dt).days + 1)
    except Exception:
        expected_days = None

    if expected_days is not None:
        actual_days = draft_itinerary.get("total_days")
        days_list = draft_itinerary.get("days", [])

        if actual_days != expected_days:
            issues.append(
                f"Trip duration mismatch: itinerary specifies {actual_days} days, expected {expected_days}."
            )
        if len(days_list) != expected_days:
            issues.append(
                f"Itinerary day count mismatch: {len(days_list)} days defined, expected {expected_days}."
            )

    # 2. Verify total estimated cost is within budget
    total_budget = float(request.get("total_trip_budget", 0))
    estimated_cost = float(draft_itinerary.get("total_estimated_cost", 0))
    if estimated_cost > total_budget:
        issues.append(
            f"Budget exceeded: total estimated cost {estimated_cost} exceeds allocated budget {total_budget}."
        )

    return issues
