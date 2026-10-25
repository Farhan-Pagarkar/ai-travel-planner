"""Activity Lookup Tool for extracting and ranking verified activity candidates."""

from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse


def normalize_activity_name(name: Any) -> str:
    """Normalize activity name for strict, consistent comparison."""
    if not name:
        return ""
    cleaned = (
        str(name)
        .strip("'\"")
        .strip("'\"")
        .replace("–", "-")
        .replace("—", "-")
        .replace("’", "'")
        .replace("`", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    return " ".join(cleaned.lower().split())


def is_matching_activity_name(name1: Any, name2: Any) -> bool:
    """Compare two activity names strictly using case-insensitive whitespace normalization.

    Disallows fuzzy matching that could permit unrelated attractions to match.
    """
    norm1 = normalize_activity_name(name1)
    norm2 = normalize_activity_name(name2)
    return bool(norm1 and norm2 and norm1 == norm2)


def is_valid_candidate_url(url: Any) -> bool:
    """Validate that a candidate URL is an absolute HTTP or HTTPS URL."""
    if not url or not isinstance(url, str):
        return False
    trimmed = url.strip()
    if not trimmed:
        return False
    try:
        parsed = urlparse(trimmed)
        return parsed.scheme.lower() in ("http", "https") and bool(parsed.netloc.strip())
    except Exception:
        return False


@dataclass
class ActivityCandidate:
    """Normalized activity or event candidate grounded in research."""

    id: str
    name: str
    description: str
    category: str
    source_url: str
    event_date: Optional[str] = None
    candidate_type: str = "attraction"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "source_url": self.source_url,
            "event_date": self.event_date,
            "candidate_type": self.candidate_type,
        }


@dataclass
class ActivityLookupResult:
    """Result container returned by the Activity Lookup Tool."""

    candidates: list[ActivityCandidate] = field(default_factory=list)
    total_candidates: int = 0
    is_sufficient: bool = True
    notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "total_candidates": self.total_candidates,
            "is_sufficient": self.is_sufficient,
            "notes": self.notes,
        }


def score_candidate(candidate: ActivityCandidate, normalized_interests: list[str]) -> int:
    """Score candidate by counting keyword matches against traveler interests."""
    if not normalized_interests:
        return 0
    text = f"{candidate.name} {candidate.description} {candidate.category}".lower()
    score = 0
    for interest in normalized_interests:
        if interest in text:
            score += 2
        else:
            # Check for partial word matches (e.g. 'temple' in 'temples')
            words = interest.split()
            for w in words:
                if len(w) >= 3 and w in text:
                    score += 1
    return score


_score_candidate = score_candidate


def lookup_activity_candidates(
    research_data: Optional[dict[str, Any]],
    interests: Optional[list[str]] = None,
    limit: int = 15,
    min_required: int = 3,
) -> ActivityLookupResult:
    """Extract, filter, and rank verified activity candidates from ResearchResult data.

    Only sourced attractions and upcoming events with verified source URLs are included.
    Historical reports and general local news are excluded from activity candidates.
    """
    if not research_data or not isinstance(research_data, dict):
        return ActivityLookupResult(
            candidates=[],
            total_candidates=0,
            is_sufficient=False,
            notes="No research data available to extract activity candidates.",
        )

    candidates: list[ActivityCandidate] = []
    seen_names: set[str] = set()
    seen_urls: set[str] = set()

    # 1. Extract attractions
    raw_attractions = research_data.get("attractions", [])
    attr_idx = 1
    for item in raw_attractions:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name", "")).strip()
        source_url = item.get("source_url")

        # Exclude candidates without verified names or absolute HTTP(S) source URLs
        if not name or not is_valid_candidate_url(source_url):
            continue

        url_str = str(source_url).strip()
        norm_name = name.lower()
        if norm_name in seen_names or url_str in seen_urls:
            continue

        seen_names.add(norm_name)
        seen_urls.add(url_str)

        desc = str(item.get("description", "")).strip()
        category = "attraction"

        candidates.append(
            ActivityCandidate(
                id=f"attr-{attr_idx}",
                name=name,
                description=desc,
                category=category,
                source_url=url_str,
                event_date=None,
                candidate_type="attraction",
            )
        )
        attr_idx += 1

    # 2. Extract upcoming local events (distinct from general attractions and historical news)
    raw_events = research_data.get("upcoming_local_events", [])
    event_idx = 1
    for item in raw_events:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "")).strip()
        source_url = item.get("source_url")

        # Exclude events without verified titles or absolute HTTP(S) source URLs
        if not title or not is_valid_candidate_url(source_url):
            continue

        url_str = str(source_url).strip()
        norm_title = title.lower()
        if norm_title in seen_names or url_str in seen_urls:
            continue

        seen_names.add(norm_title)
        seen_urls.add(url_str)

        summary = str(item.get("summary", "")).strip()
        category = str(item.get("category", "event")).strip()
        event_date = item.get("event_date")

        candidates.append(
            ActivityCandidate(
                id=f"event-{event_idx}",
                name=title,
                description=summary,
                category=category or "event",
                source_url=url_str,
                event_date=str(event_date).strip() if event_date else None,
                candidate_type="upcoming_event",
            )
        )
        event_idx += 1

    # 3. Rank candidates according to traveler interests
    normalized_interests = [i.strip().lower() for i in (interests or []) if i and i.strip()]
    if normalized_interests:
        candidates.sort(
            key=lambda c: _score_candidate(c, normalized_interests),
            reverse=True,
        )

    # 4. Apply limit
    total_found = len(candidates)
    if limit > 0:
        candidates = candidates[:limit]

    # 5. Check sufficiency
    is_sufficient = len(candidates) >= min_required
    notes = None
    if not is_sufficient:
        notes = (
            f"Insufficient verified activity candidates found in research "
            f"({len(candidates)} available, minimum {min_required} recommended). "
            f"Free-time or leisure blocks should be scheduled."
        )

    return ActivityLookupResult(
        candidates=candidates,
        total_candidates=total_found,
        is_sufficient=is_sufficient,
        notes=notes,
    )
