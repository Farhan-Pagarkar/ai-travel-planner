from dataclasses import dataclass, field
from typing import Any, Optional

# Standard allocation policy percentages summing to exactly 100% (1.00)
DEFAULT_BUDGET_POLICY: dict[str, float] = {
    "accommodation": 0.35,        # 35%
    "food": 0.25,                 # 25%
    "activities": 0.20,           # 20%
    "local_transportation": 0.10, # 10%
    "contingency": 0.10,          # 10%
}


@dataclass
class BudgetAllocation:
    """Structured budget allocation with category breakdowns and constraints."""

    total_budget: float
    currency: str
    traveler_count: int
    trip_days: int
    budget_per_traveler: float
    budget_per_day: float
    breakdown: dict[str, float] = field(default_factory=dict)
    contingency_reserve: float = 0.0
    available_for_planned_expenses: float = 0.0
    daily_activity_allowance: float = 0.0
    allocation_policy_percentages: dict[str, float] = field(default_factory=dict)
    notes: str = (
        "Allocations reflect planning constraints and estimated allowances based on a "
        "deterministic budget policy. Actual costs remain estimates until real booking data is available."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_budget": self.total_budget,
            "currency": self.currency,
            "traveler_count": self.traveler_count,
            "trip_days": self.trip_days,
            "budget_per_traveler": self.budget_per_traveler,
            "budget_per_day": self.budget_per_day,
            "breakdown": self.breakdown,
            "contingency_reserve": self.contingency_reserve,
            "available_for_planned_expenses": self.available_for_planned_expenses,
            "daily_activity_allowance": self.daily_activity_allowance,
            "allocation_policy_percentages": self.allocation_policy_percentages,
            "notes": self.notes,
        }


def calculate_budget_allocation(
    total_trip_budget: float,
    currency: str = "USD",
    traveler_count: int = 1,
    trip_days: int = 1,
    policy: Optional[dict[str, float]] = None,
) -> BudgetAllocation:
    """Calculate deterministic budget breakdown and spending constraints.

    :param total_trip_budget: Total group budget (must be > 0).
    :param currency: Three-letter or standard currency code (default: USD).
    :param traveler_count: Total travelers in party (must be >= 1).
    :param trip_days: Total days of the trip (must be >= 1).
    :param policy: Optional custom policy mapping category names to decimal fractions summing to 1.0.
    :return: BudgetAllocation containing category breakdowns and spending caps.
    """
    if total_trip_budget is None or float(total_trip_budget) <= 0:
        raise ValueError("Total trip budget must be greater than zero.")
    if traveler_count is None or int(traveler_count) < 1:
        raise ValueError("Traveler count must be at least 1.")
    if trip_days is None or int(trip_days) < 1:
        raise ValueError("Trip duration in days must be at least 1.")
    if not currency or not str(currency).strip():
        currency = "USD"
    else:
        currency = str(currency).strip().upper()

    alloc_policy = dict(policy or DEFAULT_BUDGET_POLICY)
    total_percentage = round(sum(alloc_policy.values()), 4)
    if total_percentage != 1.0:
        raise ValueError(
            f"Budget allocation percentages must sum to 100% (got {round(total_percentage * 100, 2)}%)."
        )

    budget = round(float(total_trip_budget), 2)
    travelers = int(traveler_count)
    days = int(trip_days)

    # Compute raw rounded categories
    breakdown: dict[str, float] = {}
    for category, pct in alloc_policy.items():
        breakdown[category] = round(budget * pct, 2)

    # Reconcile any rounding cents to ensure sum(breakdown) == budget exactly
    diff = round(budget - sum(breakdown.values()), 2)
    if diff != 0:
        # Adjust accommodation or first available category
        adj_cat = "accommodation" if "accommodation" in breakdown else next(iter(breakdown))
        breakdown[adj_cat] = round(breakdown[adj_cat] + diff, 2)

    contingency = breakdown.get("contingency", 0.0)
    available_planned = round(budget - contingency, 2)
    activity_allowance = round(breakdown.get("activities", 0.0) / days, 2)

    return BudgetAllocation(
        total_budget=budget,
        currency=currency,
        traveler_count=travelers,
        trip_days=days,
        budget_per_traveler=round(budget / travelers, 2),
        budget_per_day=round(budget / days, 2),
        breakdown=breakdown,
        contingency_reserve=contingency,
        available_for_planned_expenses=available_planned,
        daily_activity_allowance=activity_allowance,
        allocation_policy_percentages={k: round(v, 4) for k, v in alloc_policy.items()},
    )
