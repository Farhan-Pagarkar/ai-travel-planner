"""Web search tool using Serper API for Google search and news retrieval."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
import httpx

from app.config import get_serper_api_key

SERPER_SEARCH_URL = "https://google.serper.dev/search"
SERPER_NEWS_URL = "https://google.serper.dev/news"


@dataclass
class SearchResultItem:
    """Normalized search result item across web search and news."""

    title: str
    url: str
    snippet: str
    source: Optional[str] = None
    publication_date: Optional[str] = None


class SerperClient:
    """Client for executing targeted Google searches and news via Serper API."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0):
        self._api_key = api_key
        self.timeout = timeout

    def _get_key(self) -> str:
        if self._api_key:
            return self._api_key
        return get_serper_api_key()

    def search(
        self,
        query: str,
        search_type: str = "search",
        limit: int = 5,
    ) -> list[SearchResultItem]:
        """Execute a search query and return normalized results.

        :param query: The search query string.
        :param search_type: 'search' for general web search, 'news' for Google News.
        :param limit: Maximum number of results to request (default: 5).
        :return: List of SearchResultItem objects.
        """
        api_key = self._get_key()
        endpoint = SERPER_NEWS_URL if search_type == "news" else SERPER_SEARCH_URL
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "num": limit,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code in (401, 403):
                raise PermissionError("Serper API authentication failed. Verify SERPER_API_KEY.") from None
            if status_code == 429:
                raise RuntimeError("Serper API rate limit exceeded.") from None
            raise RuntimeError(f"Serper API returned HTTP {status_code}: {exc.response.text}") from None
        except httpx.TimeoutException:
            raise TimeoutError(f"Serper API request timed out after {self.timeout}s.") from None
        except httpx.RequestError as exc:
            raise RuntimeError(f"Serper API network request failed: {exc}") from None
        except Exception as exc:
            raise RuntimeError(f"Failed to parse Serper API response: {exc}") from None

        return self._normalize_results(data, search_type=search_type)

    def _normalize_results(self, data: dict[str, Any], search_type: str) -> list[SearchResultItem]:
        """Normalize raw Serper JSON into consistent SearchResultItem objects."""
        items: list[SearchResultItem] = []
        raw_items = data.get("news" if search_type == "news" else "organic", [])

        if not isinstance(raw_items, list):
            return items

        for raw in raw_items:
            if not isinstance(raw, dict):
                continue

            title = raw.get("title") or "Untitled"
            url = raw.get("link") or ""
            snippet = raw.get("snippet") or ""
            source = raw.get("source") or None
            pub_date = raw.get("date") or None

            items.append(
                SearchResultItem(
                    title=str(title).strip(),
                    url=str(url).strip(),
                    snippet=str(snippet).strip(),
                    source=str(source).strip() if source else None,
                    publication_date=str(pub_date).strip() if pub_date else None,
                )
            )

        return items


def gather_destination_research(
    destination: str,
    start_date: str,
    end_date: str,
    interests: list[str],
    client: Optional[SerperClient] = None,
) -> dict[str, list[SearchResultItem]]:
    """Gather targeted destination, news, events, and historical evidence.

    Uses 4 targeted queries to avoid excessive search credit usage while
    covering all required research aspects.
    """
    if client is None:
        client = SerperClient()

    interests_text = " ".join(interests[:3]) if interests else "attractions"

    # 1. Destination Overview, Attractions, Safety, Tips
    general_query = f"{destination} travel guide top attractions visitor tips safety {interests_text}"

    # 2. Recent Local News & Advisories (closures, transport disruptions)
    news_query = f"{destination} local news travel advisory transport disruption closure"

    # 3. Upcoming Scheduled Events & Festivals during travel dates
    events_query = f"{destination} events festivals exhibitions {start_date} to {end_date}"

    # 4. Historical Local Reports from the corresponding period in previous year
    try:
        start_year = datetime.strptime(start_date, "%Y-%m-%d").year
        prev_year = start_year - 1
        month_name = datetime.strptime(start_date, "%Y-%m-%d").strftime("%B")
    except Exception:
        prev_year = datetime.now().year - 1
        month_name = ""

    historical_query = f"{destination} weather disruptions events {month_name} {prev_year}"

    results: dict[str, list[SearchResultItem]] = {
        "destination_info": [],
        "recent_news": [],
        "upcoming_events": [],
        "historical_reports": [],
    }

    # Execute searches safely
    try:
        results["destination_info"] = client.search(general_query, search_type="search", limit=4)
    except Exception as exc:
        results["destination_info"] = [
            SearchResultItem(
                title="Search Error",
                url="",
                snippet=f"Destination search unavailable: {exc}",
            )
        ]

    try:
        results["recent_news"] = client.search(news_query, search_type="news", limit=4)
    except Exception as exc:
        results["recent_news"] = []

    try:
        results["upcoming_events"] = client.search(events_query, search_type="search", limit=4)
    except Exception as exc:
        results["upcoming_events"] = []

    try:
        results["historical_reports"] = client.search(historical_query, search_type="search", limit=4)
    except Exception as exc:
        results["historical_reports"] = []

    return results
