"""Research Agent placeholder module.

Performs destination research, attractions lookup, and weather forecasting.
Real web search and weather API integrations will be added in subsequent phases.
"""

from typing import Any


def run_researcher_placeholder(request: dict[str, Any]) -> dict[str, Any]:
    """Generate deterministic destination research findings.

    NOTE: External web search and weather APIs will be added later.
    """
    destination = request.get("destination", "Unknown")
    interests = request.get("interests", ["sightseeing"])

    return {
        "destination": destination,
        "weather_summary": f"Mild and seasonal weather conditions expected for {destination}.",
        "top_attractions": [
            f"{destination} Cultural Heritage Center",
            f"{destination} National Park & Scenic Viewpoint",
            f"{destination} Historic Market",
        ],
        "interest_matches": [f"Recommended activities for {interest}" for interest in interests],
        "travel_tips": [
            "Opt for central accommodations to minimize commute times.",
            "Pre-book admission tickets for peak attractions.",
        ],
        "notes": "Placeholder research data. Real web search and weather integrations will be added later.",
    }
