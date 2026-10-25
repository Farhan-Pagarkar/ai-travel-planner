"""Itinerary Planner Agent module utilizing Groq, Activity Lookup Tool, and Budget Calculator Tool."""

from datetime import date, datetime, timedelta
import json
import math
import re
from typing import Any, Optional
from groq import Groq

from app.config import get_groq_api_key, get_groq_model
from app.schemas.itinerary import Itinerary
from app.schemas.research import ResearchResult
from app.tools.activities import (
    ActivityCandidate,
    is_matching_activity_name,
    lookup_activity_candidates,
    normalize_activity_name,
)
from app.tools.budget import calculate_budget_allocation


class PlannerError(RuntimeError):
    """Raised when itinerary planning fails due to generation, provider, or validation errors."""
    pass


def _sanitize_error_message(msg: str) -> str:
    """Scrub sensitive credentials, API keys, or authorization headers from error messages."""
    if not msg:
        return ""
    scrubbed = re.sub(r"gsk_[A-Za-z0-9_-]+", "[REDACTED_API_KEY]", str(msg))
    scrubbed = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED]", scrubbed, flags=re.IGNORECASE)
    scrubbed = re.sub(
        r"['\"]?Authorization['\"]?\s*:\s*['\"][^'\"]+['\"]",
        "'Authorization': '[REDACTED]'",
        scrubbed,
        flags=re.IGNORECASE,
    )
    return scrubbed


def _extract_and_parse_json(content: str) -> dict[str, Any]:
    """Parse JSON content safely, stripping markdown fences or wrapping whitespace if present."""
    if not content or not content.strip():
        raise ValueError("Empty response received from itinerary generator.")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def _parse_safe_float(val: Any) -> Optional[float]:
    """Parse a float safely, rejecting None, booleans, non-numbers, NaN, and infinity."""
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


PLANNER_SYSTEM_PROMPT = """You are an expert, meticulous travel itinerary planner.
Your task is to construct a structured, day-by-day travel itinerary grounded strictly in the provided research, activity candidates, and budget constraints.

CRITICAL GROUNDING AND VALIDATION RULES:
1. Cover every single calendar date from start_date to end_date inclusively, in exact chronological order without gaps or duplicate dates.
2. For all named attractions and upcoming events, you MUST select a candidate from the provided CANDIDATE ACTIVITIES list.
3. For each selected candidate, copy its exact "id" into candidate_id, its exact "source_url" into source_url, and use its exact candidate name for the activity name.
4. NEVER invent attraction names, venue names, event dates, candidate IDs, or source URLs. Never use a candidate ID/URL with an unrelated activity name.
5. Generic meals, rest periods, transit, and free-time blocks must have candidate_id: null and source_url: null. Do not invent restaurant names or claim generic activities are verified venues.
6. The sum of all activity estimated_cost values for a day must equal that day's estimated_cost.
7. The sum of all daily estimated_cost values must equal total_estimated_cost.
8. total_estimated_cost must not exceed the total trip budget.
9. All costs must be non-negative numbers.
10. If candidate activities are limited, schedule leisure, free-time, or scenic walking blocks rather than inventing attractions. Note any limitations in warnings.
11. If user feedback or requested modifications are provided, incorporate them into the plan while keeping unaffected days consistent.
12. Return ONLY a single valid JSON object strictly matching the schema. Do not output markdown code fences, think tags, or extra commentary.

SCHEMA SPECIFICATION:
{
  "destination": "string",
  "total_days": 1,
  "currency": "USD",
  "total_estimated_cost": 250.0,
  "days": [
    {
      "day": 1,
      "date": "YYYY-MM-DD",
      "focus": "string",
      "theme": "string or null",
      "activities": [
        {
          "name": "string",
          "description": "string",
          "category": "string",
          "start_time": "string or null",
          "estimated_duration": "string or null",
          "estimated_cost": 25.0,
          "candidate_id": "candidate_id or null",
          "source_url": "https://... or null"
        }
      ],
      "estimated_cost": 250.0
    }
  ],
  "summary": "string",
  "assumptions": ["string"],
  "warnings": ["string"]
}
"""


def _parse_date(date_val: Any) -> date:
    if isinstance(date_val, date):
        return date_val
    return datetime.strptime(str(date_val), "%Y-%m-%d").date()


def _format_candidate_activities_prompt(candidates: list[ActivityCandidate]) -> str:
    """Format candidate records into structured text for LLM grounding."""
    if not candidates:
        return "NO VERIFIED CANDIDATE ACTIVITIES AVAILABLE FROM RESEARCH. (Schedule leisure, free exploration, and scenic walks with candidate_id: null)."

    lines = [f"Found {len(candidates)} verified candidate activities:"]
    for c in candidates:
        event_info = f" | Event Date: {c.event_date}" if c.event_date else ""
        lines.append(
            f"- [ID: {c.id}] Name: {c.name} | Type: {c.candidate_type} | Category: {c.category}{event_info}\n"
            f"  Description: {c.description}\n"
            f"  Source URL: {c.source_url}"
        )
    return "\n".join(lines)


def _format_budget_prompt(budget_alloc: Any) -> str:
    """Format budget breakdown and constraints for prompt."""
    b = budget_alloc.to_dict()
    breakdown_lines = [f"  - {cat}: {amount} {b['currency']}" for cat, amount in b["breakdown"].items()]
    return (
        f"TOTAL GROUP BUDGET: {b['total_budget']} {b['currency']}\n"
        f"BUDGET PER TRAVELER: {b['budget_per_traveler']} {b['currency']} ({b['traveler_count']} traveler(s))\n"
        f"DAILY GROUP BUDGET CAP: {b['budget_per_day']} {b['currency']} ({b['trip_days']} day(s))\n"
        f"PLANNED EXPENSE ALLOWANCE (AFTER {b['contingency_reserve']} {b['currency']} CONTINGENCY): {b['available_for_planned_expenses']} {b['currency']}\n"
        f"CATEGORY ALLOCATIONS (ESTIMATES):\n" + "\n".join(breakdown_lines) + "\n"
        f"NOTE: {b['notes']}"
    )


def _validate_itinerary_grounding(
    itinerary_data: dict[str, Any],
    expected_dates: list[str],
    max_budget: float,
    valid_candidates: dict[str, ActivityCandidate],
    destination: Optional[str] = None,
    currency: Optional[str] = None,
) -> list[str]:
    """Perform deterministic grounding and math checks on synthesized itinerary.

    Returns list of error messages, or empty list if valid.
    """
    errors: list[str] = []

    if not isinstance(itinerary_data, dict):
        return ["Itinerary data must be a JSON object."]

    # Destination and currency checks
    if destination is not None:
        itin_dest = str(itinerary_data.get("destination", "")).strip()
        if not itin_dest:
            errors.append("Itinerary destination is missing.")
        elif itin_dest.lower() != destination.strip().lower():
            errors.append(
                f"Itinerary destination '{itin_dest}' does not match requested destination '{destination.strip()}'."
            )

    if currency is not None:
        itin_curr = str(itinerary_data.get("currency", "")).strip().upper()
        if not itin_curr:
            errors.append("Itinerary currency is missing.")
        elif itin_curr != currency.strip().upper():
            errors.append(
                f"Itinerary currency '{itin_curr}' does not match requested currency '{currency.strip().upper()}'."
            )

    # 1. Date coverage and order checks
    days = itinerary_data.get("days", [])
    if not isinstance(days, list) or len(days) != len(expected_dates):
        errors.append(
            f"Itinerary day count mismatch: expected {len(expected_dates)} day(s), got {len(days) if isinstance(days, list) else 0}."
        )
        return errors

    actual_dates = [d.get("date") for d in days if isinstance(d, dict)]
    if actual_dates != expected_dates:
        errors.append(f"Itinerary dates {actual_dates} do not match expected chronological dates {expected_dates}.")

    if len(set(actual_dates)) != len(actual_dates):
        errors.append("Duplicate day dates detected in itinerary.")

    for d in days:
        if isinstance(d, dict):
            raw_d = d.get("date")
            if not raw_d or not isinstance(raw_d, str):
                errors.append(f"Day {d.get('day', '?')} has malformed or missing date: {raw_d}.")
            else:
                try:
                    date.fromisoformat(raw_d.strip())
                except Exception:
                    errors.append(f"Day {d.get('day', '?')} has malformed date format: {raw_d}.")

    GENERIC_CATEGORIES = {
        "meal", "dining", "food", "lunch", "dinner", "breakfast",
        "leisure", "transit", "transportation", "travel", "rest",
        "free_time", "relaxation", "scenic_walk", "walk"
    }
    ATTRACTION_EVENT_CATEGORIES = {
        "attraction", "event", "upcoming_event", "sightseeing",
        "cultural_event", "festival", "monument", "museum", "temple", "shrine", "tour"
    }

    # 2. Activity candidate grounding and URL preservation
    for day in days:
        if not isinstance(day, dict):
            errors.append("Encountered non-dictionary day entry in itinerary.")
            continue
        day_num = day.get("day", "?")
        activities = day.get("activities", [])
        if not isinstance(activities, list):
            errors.append(f"Day {day_num} activities must be a list.")
            continue

        for act in activities:
            if not isinstance(act, dict):
                errors.append(f"Day {day_num} contains a non-dictionary activity.")
                continue
            act_name = str(act.get("name", "Unnamed")).strip()
            cand_id = act.get("candidate_id")
            if cand_id is not None:
                cand_id = str(cand_id).strip()
                if not cand_id:
                    cand_id = None
            source_url = act.get("source_url")
            if source_url is not None:
                source_url = str(source_url).strip()
                if not source_url:
                    source_url = None

            raw_cost = act.get("estimated_cost", 0.0)
            parsed_cost = _parse_safe_float(raw_cost)
            if parsed_cost is None or parsed_cost < 0:
                errors.append(
                    f"Activity '{act_name}' on Day {day_num} has invalid cost ({raw_cost}). Must be a finite, non-negative number."
                )

            category = str(act.get("category", "")).lower().strip()

            if cand_id:
                if cand_id not in valid_candidates:
                    errors.append(
                        f"Activity '{act_name}' on Day {day_num} references unknown candidate ID '{cand_id}'. Must select from provided candidate IDs."
                    )
                else:
                    cand = valid_candidates[cand_id]
                    expected_url = cand.source_url
                    if source_url != expected_url:
                        errors.append(
                            f"Activity '{act_name}' on Day {day_num} has source URL '{source_url}' that does not match candidate '{cand_id}' URL '{expected_url}'."
                        )
                    if not is_matching_activity_name(act_name, cand.name):
                        errors.append(
                            f"Activity '{act_name}' on Day {day_num} does not match candidate '{cand_id}' name '{cand.name}'."
                        )
            else:
                if source_url:
                    errors.append(
                        f"Generic activity '{act_name}' on Day {day_num} provides a source URL without a verified candidate ID."
                    )
                is_attraction_or_event = (
                    category in ATTRACTION_EVENT_CATEGORIES
                    or category.startswith("attraction")
                    or category.startswith("event")
                    or act.get("candidate_type") in {"attraction", "upcoming_event"}
                )
                if is_attraction_or_event and category not in GENERIC_CATEGORIES:
                    errors.append(
                        f"Activity '{act_name}' on Day {day_num} is classified as an attraction/event ({category}) but is missing a verified candidate ID and source URL."
                    )

    # 3. Cost arithmetic checks
    daily_cost_sum = 0.0
    for day in days:
        if not isinstance(day, dict):
            continue
        day_num = day.get("day", "?")
        activities = day.get("activities", [])
        act_cost_sum = 0.0
        if isinstance(activities, list):
            for a in activities:
                if isinstance(a, dict):
                    c = _parse_safe_float(a.get("estimated_cost", 0.0))
                    if c is not None and c >= 0:
                        act_cost_sum = round(act_cost_sum + round(c, 2), 2)

        raw_day_cost = day.get("estimated_cost", 0.0)
        parsed_day_cost = _parse_safe_float(raw_day_cost)
        if parsed_day_cost is None or parsed_day_cost < 0:
            errors.append(f"Day {day_num} has invalid estimated_cost ({raw_day_cost}).")
            parsed_day_cost = 0.0
        else:
            parsed_day_cost = round(parsed_day_cost, 2)

        if abs(act_cost_sum - parsed_day_cost) > 0.05:
            errors.append(
                f"Day {day_num} activity costs sum to {act_cost_sum}, which does not match reported day estimated_cost {parsed_day_cost}."
            )
        daily_cost_sum = round(daily_cost_sum + parsed_day_cost, 2)

    raw_total_cost = itinerary_data.get("total_estimated_cost", 0.0)
    parsed_total_cost = _parse_safe_float(raw_total_cost)
    if parsed_total_cost is None or parsed_total_cost < 0:
        errors.append(f"Total estimated cost has invalid value ({raw_total_cost}).")
        parsed_total_cost = 0.0
    else:
        parsed_total_cost = round(parsed_total_cost, 2)

    if abs(daily_cost_sum - parsed_total_cost) > 0.05:
        errors.append(
            f"Total estimated cost ({parsed_total_cost}) does not match sum of daily costs ({daily_cost_sum})."
        )

    if parsed_total_cost > max_budget:
        errors.append(
            f"Total estimated cost ({parsed_total_cost}) exceeds total trip budget ({max_budget})."
        )

    return errors


def run_planner_agent(
    request: dict[str, Any],
    research: Optional[dict[str, Any]] = None,
    feedback: Optional[str] = None,
    modifications: Optional[str] = None,
    revision_count: int = 0,
    groq_client: Optional[Groq] = None,
    groq_model: Optional[str] = None,
) -> dict[str, Any]:
    """Synthesize a complete, research-grounded day-by-day itinerary."""
    destination = str(request.get("destination", "")).strip()
    start_date_str = str(request.get("start_date", "")).strip()
    end_date_str = str(request.get("end_date", "")).strip()
    interests = request.get("interests", ["sightseeing"])
    currency = str(request.get("currency", "USD")).strip().upper()
    total_budget = float(request.get("total_trip_budget", 1000.0))
    traveler_count = int(request.get("traveler_count", 1))

    if not destination:
        raise ValueError("Travel request missing required 'destination'.")
    if not start_date_str or not end_date_str:
        raise ValueError("Travel request missing travel dates.")

    start_dt = _parse_date(start_date_str)
    end_dt = _parse_date(end_date_str)
    if end_dt < start_dt:
        raise ValueError(f"Trip end_date {end_date_str} cannot precede start_date {start_date_str}.")

    # 1. Build expected chronological dates list
    expected_dates: list[str] = []
    curr = start_dt
    while curr <= end_dt:
        expected_dates.append(curr.isoformat())
        curr += timedelta(days=1)
    num_days = len(expected_dates)

    # 2. Normalize and validate research data
    normalized_research: dict[str, Any] = {}
    if research and isinstance(research, dict):
        try:
            validated_res = ResearchResult.model_validate(research)
            normalized_research = validated_res.model_dump(mode="json")
        except Exception:
            normalized_research = research

    # 3. Call Tool 1: Activity Lookup Tool
    activity_lookup = lookup_activity_candidates(
        research_data=normalized_research,
        interests=interests,
        limit=20,
        min_required=num_days,
    )
    valid_candidates = {c.id: c for c in activity_lookup.candidates}

    # 4. Call Tool 2: Budget Calculator Tool
    budget_allocation = calculate_budget_allocation(
        total_trip_budget=total_budget,
        currency=currency,
        traveler_count=traveler_count,
        trip_days=num_days,
    )

    # 5. Initialize Groq client
    if groq_client is None:
        groq_api_key = get_groq_api_key()
        groq_client = Groq(api_key=groq_api_key)

    if groq_model is None:
        groq_model = get_groq_model()

    # 6. Build LLM prompt
    candidates_text = _format_candidate_activities_prompt(activity_lookup.candidates)
    budget_text = _format_budget_prompt(budget_allocation)

    revision_context = ""
    if modifications:
        revision_context = f"\nUSER REQUESTED MODIFICATIONS (Revision {revision_count}):\n{modifications}\nApply these changes while preserving unaffected days."
    elif feedback:
        revision_context = f"\nUSER REVIEW FEEDBACK (Revision {revision_count}):\n{feedback}\nAddress this feedback while keeping unaffected days consistent."

    weather_notes = ""
    if normalized_research.get("weather"):
        weather_notes = f"\nDESTINATION WEATHER: {normalized_research['weather'].get('notes', 'Normal conditions.')}"

    user_prompt = (
        f"Create a {num_days}-day itinerary for {destination} from {start_date_str} to {end_date_str}.\n\n"
        f"TRAVEL DATES REQUIRED ({num_days} days): {', '.join(expected_dates)}\n"
        f"TRAVELER INTERESTS: {', '.join(interests)}\n"
        f"{weather_notes}\n\n"
        f"--- BUDGET CONSTRAINTS ---\n{budget_text}\n\n"
        f"--- CANDIDATE ACTIVITIES (SELECT FROM THESE) ---\n{candidates_text}\n"
        f"{revision_context}\n\n"
        "Generate strictly valid JSON conforming to the schema. "
        "Use candidate_id and exact source_url for all researched attractions/events. "
        "Use candidate_id: null and source_url: null for generic meals and free time. "
        "Ensure daily costs sum to total_estimated_cost and do not exceed total group budget."
    )

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    groq_call_kwargs: dict[str, Any] = {
        "model": groq_model,
        "response_format": {"type": "json_object"},
        "reasoning_effort": "low",
        "include_reasoning": False,
        "max_tokens": 4096,
        "temperature": 0.1,
    }

    # Attempt 1: Generation
    content = ""
    parsed_itinerary: Optional[dict[str, Any]] = None
    initial_error: Optional[Exception] = None

    try:
        completion = groq_client.chat.completions.create(
            messages=messages,
            **groq_call_kwargs,
        )
        content = completion.choices[0].message.content or ""
        parsed_itinerary = _extract_and_parse_json(content)
    except Exception as exc:
        initial_error = exc

    # Bounded single correction attempt if initial generation call failed (e.g., json_validate_failed)
    if initial_error is not None:
        sanitized_init_err = _sanitize_error_message(str(initial_error))
        correction_prompt = (
            f"Your previous attempt to generate the itinerary failed: {sanitized_init_err}.\n"
            "Please generate strictly valid JSON conforming exactly to the schema specification. "
            "Return ONLY the single JSON object without markdown fences, think blocks, or extra commentary."
        )
        messages.append({"role": "user", "content": correction_prompt})
        try:
            correction_completion = groq_client.chat.completions.create(
                messages=messages,
                **groq_call_kwargs,
            )
            content = correction_completion.choices[0].message.content or ""
            parsed_itinerary = _extract_and_parse_json(content)
        except Exception as exc:
            sanitized_exc = _sanitize_error_message(str(exc))
            raise PlannerError(f"Groq itinerary generation failed after retry: {sanitized_exc}") from None

    # Deterministic grounding verification
    errors = _validate_itinerary_grounding(
        itinerary_data=parsed_itinerary,
        expected_dates=expected_dates,
        max_budget=total_budget,
        valid_candidates=valid_candidates,
        destination=destination,
        currency=currency,
    )

    # Bounded single correction attempt if validation failed AND we haven't already used our single retry
    if errors and initial_error is None:
        correction_prompt = (
            f"Your previous itinerary output had the following validation errors:\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\n\nPlease correct these errors and regenerate the complete, strictly valid JSON itinerary. "
            "Ensure exact dates, valid candidate IDs and URLs, and exact cost math."
        )
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": correction_prompt})

        try:
            correction_completion = groq_client.chat.completions.create(
                messages=messages,
                **groq_call_kwargs,
            )
            content = correction_completion.choices[0].message.content or ""
            parsed_itinerary = _extract_and_parse_json(content)
        except Exception as exc:
            sanitized_exc = _sanitize_error_message(str(exc))
            raise PlannerError(f"Groq itinerary correction attempt failed: {sanitized_exc}") from None

        # Re-validate after correction attempt
        errors = _validate_itinerary_grounding(
            itinerary_data=parsed_itinerary,
            expected_dates=expected_dates,
            max_budget=total_budget,
            valid_candidates=valid_candidates,
            destination=destination,
            currency=currency,
        )

    if errors:
        raise PlannerError(
            f"Itinerary planning failed validation after correction attempt: {'; '.join(errors)}"
        )

    # 7. Post-process itinerary metadata
    parsed_itinerary["destination"] = destination
    parsed_itinerary["total_days"] = num_days
    parsed_itinerary["currency"] = currency
    parsed_itinerary["budget_breakdown"] = budget_allocation.to_dict()

    if revision_count > 0 and "Revision" not in parsed_itinerary.get("summary", ""):
        parsed_itinerary["summary"] = f"{parsed_itinerary.get('summary', '')} (Revision {revision_count})"

    if not activity_lookup.is_sufficient and activity_lookup.notes:
        warnings_list = parsed_itinerary.setdefault("warnings", [])
        if activity_lookup.notes not in warnings_list:
            warnings_list.append(activity_lookup.notes)

    # 8. Final Pydantic validation
    try:
        validated = Itinerary.model_validate(parsed_itinerary)
    except Exception as exc:
        raise PlannerError(f"Itinerary schema validation failed: {exc}") from None

    return validated.model_dump(mode="json")


# Backwards compatibility alias
run_planner_placeholder = run_planner_agent
