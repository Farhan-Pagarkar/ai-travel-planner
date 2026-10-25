import json
from typing import Any, Optional
from groq import Groq

from app.config import get_groq_api_key, get_groq_model, get_serper_api_key
from app.schemas.research import ResearchResult, WeatherContext
from app.tools.weather import WeatherClient, gather_weather_intelligence
from app.tools.web_search import SearchEvidence, SerperClient, gather_destination_research

SYSTEM_PROMPT = """You are an expert, meticulous travel intelligence researcher.
Your task is to synthesize the provided destination search results and weather data into a strict JSON object.

CRITICAL GROUNDING RULES:
1. Ground all findings strictly in the provided search evidence.
2. NEVER invent URLs, citations, publication dates, event dates, or weather figures.
3. If a publication date or event date is not in the evidence, record it as null.
4. Separate recent local news from historical reports (from previous years) and scheduled upcoming events.
5. If an event or disruption is not confirmed by a reliable source, set is_verified to false.
6. When evidence is insufficient for any section, preserve an empty list [] and note the limitation in warnings_and_uncertainties.
7. Return ONLY a single valid JSON object matching the requested schema.

SCHEMA SPECIFICATION:
{
  "destination": "string",
  "attractions": [
    {"name": "string", "description": "string", "visitor_tips": "string or null", "source_url": "string or null"}
  ],
  "local_tips": ["string"],
  "safety_considerations": ["string"],
  "seasonal_considerations": ["string"],
  "recent_local_news": [
    {
      "title": "string",
      "summary": "string",
      "source_url": "string or null",
      "publication_date": "string or null",
      "event_date": "string or null",
      "category": "string",
      "potential_travel_impact": "string",
      "is_verified": true
    }
  ],
  "historical_local_reports": [
    {
      "title": "string",
      "summary": "string",
      "source_url": "string or null",
      "publication_date": "string or null",
      "event_date": "string or null",
      "category": "string",
      "potential_travel_impact": "string",
      "is_verified": true
    }
  ],
  "upcoming_local_events": [
    {
      "title": "string",
      "summary": "string",
      "source_url": "string or null",
      "publication_date": "string or null",
      "event_date": "string or null",
      "category": "string",
      "potential_travel_impact": "string",
      "is_verified": true
    }
  ],
  "source_urls": ["string"],
  "warnings_and_uncertainties": ["string"]
}
"""


def _format_weather_evidence(
    weather: Optional[WeatherContext],
    weather_summary: Optional[str] = None,
) -> str:
    """Format full structured WeatherContext into detailed evidence text for the LLM."""
    if weather is None:
        return f"WEATHER STATUS / SUMMARY: {weather_summary or 'Weather information unavailable.'}"

    lines: list[str] = [
        f"WEATHER TYPE / STATUS: {weather.weather_type}",
    ]
    if weather.notes:
        lines.append(f"NOTES: {weather.notes}")

    if weather.forecast_days:
        lines.append("DAILY FORECAST OBSERVATIONS:")
        for day in weather.forecast_days:
            prob_str = (
                f", Precip Probability: {day.precipitation_probability_pct}%"
                if day.precipitation_probability_pct is not None
                else ""
            )
            lines.append(
                f"- Date: {day.date} | High: {day.temp_max_c}°C | Low: {day.temp_min_c}°C | "
                f"Rain Sum: {day.precipitation_sum_mm}mm{prob_str} | Condition: {day.weather_condition}"
            )

    if weather.historical_comparison:
        hc = weather.historical_comparison
        lines.append("HISTORICAL CLIMATE COMPARISON FIGURES:")
        if hc.comparison_years:
            lines.append(f"- Comparison Years: {hc.comparison_years}")
        if hc.equivalent_dates_last_year:
            lines.append(f"- Equivalent Dates Last Year: {hc.equivalent_dates_last_year}")
        if hc.typical_temp_max_c is not None:
            lines.append(f"- Typical High: {hc.typical_temp_max_c}°C")
        if hc.typical_temp_min_c is not None:
            lines.append(f"- Typical Low: {hc.typical_temp_min_c}°C")
        if hc.typical_precipitation_mm is not None:
            lines.append(f"- Typical Total Precipitation: {hc.typical_precipitation_mm}mm")
        if hc.rainy_days_count is not None:
            lines.append(f"- Typical Rainy Days Count: {hc.rainy_days_count}")
        if hc.summary:
            lines.append(f"- Summary: {hc.summary}")

    return "\n".join(lines)


def _format_evidence_package(
    destination: str,
    start_date: str,
    end_date: str,
    interests: list[str],
    search_data: Any,
    weather_context: Optional[WeatherContext] = None,
    weather_summary: Optional[str] = None,
) -> str:
    """Format collected search items and structured weather into a prompt for the LLM."""
    weather_text = _format_weather_evidence(weather_context, weather_summary)
    lines: list[str] = [
        f"DESTINATION: {destination}",
        f"TRAVEL DATES: {start_date} to {end_date}",
        f"INTERESTS: {', '.join(interests) if interests else 'General sightseeing'}",
        "",
        "--- EVIDENCE SECTION: WEATHER INTELLIGENCE & OBSERVATIONS ---",
        weather_text,
        "",
        "--- EVIDENCE SECTION 1: DESTINATION & ATTRACTIONS ---",
    ]
    for item in search_data.get("destination_info", [])[:4]:
        lines.append(f"- Title: {item.title}\n  URL: {item.url}\n  Snippet: {item.snippet}")

    lines.append("\n--- EVIDENCE SECTION 2: RECENT LOCAL NEWS & ADVISORIES ---")
    for item in search_data.get("recent_news", [])[:4]:
        lines.append(
            f"- Title: {item.title}\n  URL: {item.url}\n  Date: {item.publication_date}\n  Source: {item.source}\n  Snippet: {item.snippet}"
        )

    lines.append("\n--- EVIDENCE SECTION 3: UPCOMING LOCAL EVENTS ---")
    for item in search_data.get("upcoming_events", [])[:4]:
        lines.append(
            f"- Title: {item.title}\n  URL: {item.url}\n  Date: {item.publication_date}\n  Snippet: {item.snippet}"
        )

    lines.append("\n--- EVIDENCE SECTION 4: HISTORICAL LOCAL REPORTS (PREVIOUS YEARS) ---")
    for item in search_data.get("historical_local_reports", [])[:4]:
        lines.append(
            f"- Title: {item.title}\n  URL: {item.url}\n  Date: {item.publication_date}\n  Snippet: {item.snippet}"
        )

    return "\n".join(lines)


def _verify_source_provenance(
    parsed_json: dict[str, Any],
    valid_source_urls: set[str],
) -> None:
    """Validate source URLs attached to attractions and reports against genuine search results.

    A URL supplied by the LLM must match a URL actually returned by the search tool
    before it is retained as a source URL. If it does not match, remove the unsupported
    URL (set to None) and add a warning.

    Genuine discovered URLs are retained. Items without source URLs remain None without warning.
    """
    warnings = parsed_json.setdefault("warnings_and_uncertainties", [])

    # 1. Attractions
    for item in parsed_json.get("attractions", []):
        url = item.get("source_url")
        if url:
            if url not in valid_source_urls:
                name = item.get("name", "Attraction")
                item["source_url"] = None
                warnings.append(
                    f"Removed unsupported source URL '{url}' from attraction '{name}'."
                )
        else:
            item["source_url"] = None

    # 2. Local report categories
    sections = [
        ("recent_local_news", "recent local news"),
        ("historical_local_reports", "historical local report"),
        ("upcoming_local_events", "upcoming local event"),
    ]
    for section_key, label in sections:
        for item in parsed_json.get(section_key, []):
            url = item.get("source_url")
            if url:
                if url not in valid_source_urls:
                    title = item.get("title", "Item")
                    item["source_url"] = None
                    warnings.append(
                        f"Removed unsupported source URL '{url}' from {label} '{title}'."
                    )
            else:
                item["source_url"] = None

    # 3. Top-level source_urls
    llm_source_urls = parsed_json.get("source_urls", [])
    for top_url in llm_source_urls:
        if top_url and top_url not in valid_source_urls:
            warnings.append(
                f"Removed unsupported top-level source URL '{top_url}' not found in search results."
            )

    # Retain all genuine discovered source URLs
    parsed_json["source_urls"] = sorted(list(valid_source_urls))


def run_researcher_agent(
    request: dict[str, Any],
    serper_client: Optional[SerperClient] = None,
    weather_client: Optional[WeatherClient] = None,
    groq_client: Optional[Groq] = None,
    groq_model: Optional[str] = None,
) -> dict[str, Any]:
    """Execute live research for a travel request using Serper, Open-Meteo, and Groq."""
    destination = str(request.get("destination", "")).strip()
    start_date = str(request.get("start_date", "")).strip()
    end_date = str(request.get("end_date", "")).strip()
    interests = request.get("interests", [])

    if not destination:
        raise ValueError("Travel request missing required 'destination'.")
    if not start_date or not end_date:
        raise ValueError("Travel request missing required travel dates.")

    # 1. Verify credentials upfront to fail with clear messages if unconfigured
    if serper_client is None:
        get_serper_api_key()
        serper_client = SerperClient()

    if groq_client is None:
        groq_api_key = get_groq_api_key()
        groq_client = Groq(api_key=groq_api_key)

    if groq_model is None:
        groq_model = get_groq_model()

    if weather_client is None:
        weather_client = WeatherClient()

    # 2. Gather Web Evidence via Serper
    search_data = gather_destination_research(
        destination=destination,
        start_date=start_date,
        end_date=end_date,
        interests=interests,
        client=serper_client,
    )

    search_failures: dict[str, str] = getattr(search_data, "failures", {})
    if isinstance(search_data, SearchEvidence):
        has_results = search_data.has_any_results
        discovered_urls = {item.url for item in search_data.all_items() if item.url}
    else:
        has_results = any(bool(v) for v in search_data.values() if isinstance(v, list))
        discovered_urls = {
            item.url
            for group in search_data.values()
            if isinstance(group, list)
            for item in group
            if getattr(item, "url", None)
        }

    # Complete search failure: return clear research failure rather than fabricating
    if not has_results:
        failure_summary = (
            "; ".join(f"{cat}: {err}" for cat, err in search_failures.items())
            if search_failures
            else "No search results returned for any category."
        )
        raise RuntimeError(
            f"Destination research failed: Unable to gather search evidence for '{destination}'. "
            f"Search failures: {failure_summary}"
        )

    # 3. Gather Weather Intelligence via Open-Meteo
    coords, weather_context = gather_weather_intelligence(
        destination=destination,
        start_date_str=start_date,
        end_date_str=end_date,
        client=weather_client,
    )

    # 4. Synthesize Evidence with Groq
    weather_note = weather_context.notes or "Weather information collected."
    evidence_text = _format_evidence_package(
        destination=destination,
        start_date=start_date,
        end_date=end_date,
        interests=interests,
        search_data=search_data,
        weather_context=weather_context,
        weather_summary=weather_note,
    )

    user_prompt = (
        f"Synthesize the following research evidence for a trip to {destination} ({start_date} to {end_date}).\n\n"
        f"{evidence_text}\n\n"
        "Output strictly valid JSON conforming to the schema."
    )

    try:
        completion = groq_client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=4096,
            temperature=0.2,
        )
        choices = getattr(completion, "choices", None)
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            return {}
        content = choices[0].message.content or "{}"
        parsed_json = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Groq research synthesis returned invalid JSON: {exc}") from None
    except Exception as exc:
        raise RuntimeError(f"Groq research synthesis failed: {exc}") from None

    # 5. Inject deterministic Weather Context and Coordinates directly from Open-Meteo
    parsed_json["destination"] = destination
    parsed_json["coordinates"] = coords.model_dump() if coords else None
    parsed_json["weather"] = weather_context.model_dump()

    # 6. Record category search failures in warnings if partial failure occurred
    for cat, err in search_failures.items():
        cat_name = cat.replace("_", " ")
        warning_msg = f"Search evidence was unavailable for {cat_name}: {err}"
        if warning_msg not in parsed_json.setdefault("warnings_and_uncertainties", []):
            parsed_json["warnings_and_uncertainties"].append(warning_msg)

    # 7. Validate source provenance (verify URLs against discovered results, remove fabricated, preserve genuine)
    _verify_source_provenance(parsed_json, discovered_urls)

    # 8. Validate with Pydantic
    try:
        validated_result = ResearchResult.model_validate(parsed_json)
    except Exception as exc:
        raise RuntimeError(f"Research result validation error: {exc}") from None

    return validated_result.model_dump(mode="json")


# Alias for backwards compatibility
run_researcher_placeholder = run_researcher_agent
