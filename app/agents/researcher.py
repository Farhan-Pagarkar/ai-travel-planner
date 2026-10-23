"""Research Agent module utilizing Serper, Open-Meteo, and Groq."""

import json
from typing import Any, Optional
from groq import Groq

from app.config import get_groq_api_key, get_groq_model, get_serper_api_key
from app.schemas.research import ResearchResult
from app.tools.weather import WeatherClient, gather_weather_intelligence
from app.tools.web_search import SerperClient, gather_destination_research

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


def _format_evidence_package(
    destination: str,
    start_date: str,
    end_date: str,
    interests: list[str],
    search_data: dict[str, list[Any]],
    weather_summary: str,
) -> str:
    """Format collected search items into a token-efficient text prompt for the LLM."""
    lines: list[str] = [
        f"DESTINATION: {destination}",
        f"TRAVEL DATES: {start_date} to {end_date}",
        f"INTERESTS: {', '.join(interests) if interests else 'General sightseeing'}",
        f"WEATHER SUMMARY: {weather_summary}",
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
        content = completion.choices[0].message.content or "{}"
        parsed_json = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Groq research synthesis returned invalid JSON: {exc}") from None
    except Exception as exc:
        raise RuntimeError(f"Groq research synthesis failed: {exc}") from None

    # 5. Inject deterministic Weather Context and Coordinates
    parsed_json["destination"] = destination
    parsed_json["coordinates"] = coords.model_dump() if coords else None
    parsed_json["weather"] = weather_context.model_dump()

    # Consolidate all genuine URLs discovered from Serper into source_urls
    discovered_urls = {
        item.url for group in search_data.values() for item in group if item.url
    }
    existing_urls = set(parsed_json.get("source_urls", []))
    parsed_json["source_urls"] = sorted(list(existing_urls.union(discovered_urls)))

    # 6. Validate with Pydantic
    try:
        validated_result = ResearchResult.model_validate(parsed_json)
    except Exception as exc:
        raise RuntimeError(f"Research result validation error: {exc}") from None

    return validated_result.model_dump(mode="json")


# Alias for backwards compatibility
run_researcher_placeholder = run_researcher_agent
