"""
Finnhub market news client.
Docs: https://finnhub.io/docs/api/company-news
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import httpx
from ftfy import fix_text, TextFixerConfig

from config import get_settings

# Some upstream sources Finnhub aggregates from (e.g. SeekingAlpha) send
# already mojibake'd text — UTF-8 bytes re-decoded as Windows-1252 and
# re-encoded, e.g. "Friday's" becomes "Fridayâ€™s". fix_text repairs this
# in place; already-correct text round-trips unchanged. uncurl_quotes is
# turned off so we don't flatten legitimate curly quotes in text that was
# never broken to begin with.
_TEXT_FIX_CONFIG = TextFixerConfig(uncurl_quotes=False)


def _clean_text(value: str) -> str:
    return fix_text(value, config=_TEXT_FIX_CONFIG) if value else value


class FinnhubError(Exception):
    pass


def _relevance_from_recency(published_at: datetime, window_start: datetime, window_end: datetime) -> float:
    """
    Finnhub doesn't provide a relevance score, so approximate one from
    recency within the fetch window: the newest article scores ~1.0, the
    oldest in-window article scores the floor. This is a stated
    simplification, not a real relevance signal.
    """
    span = (window_end - window_start).total_seconds() or 1.0
    age = (window_end - published_at).total_seconds()
    frac = max(0.0, min(1.0, age / span))
    floor = 0.4
    return round(1.0 - frac * (1.0 - floor), 3)


async def fetch_company_news(ticker: str, days: int = 7, limit: int = 15) -> List[Dict[str, Any]]:
    """
    Fetch and rank recent company news from Finnhub's company-news endpoint.
    Returns dicts: title, source, url, summary, category, published_at
    (datetime), relevance (float), external_id (int).
    """
    api_key = get_settings().finnhub_api_key
    if not api_key:
        raise FinnhubError("FINNHUB_API_KEY is not configured")

    window_end = datetime.now()
    window_start = window_end - timedelta(days=days)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": ticker.upper(),
                "from": window_start.date().isoformat(),
                "to": window_end.date().isoformat(),
                "token": api_key,
            },
        )

    if resp.status_code != 200:
        raise FinnhubError(f"Finnhub API error {resp.status_code}: {resp.text[:200]}")

    raw = resp.json()
    if not isinstance(raw, list):
        raise FinnhubError(f"Finnhub API returned unexpected payload: {raw}")

    articles = []
    for r in raw:
        # Finnhub occasionally returns junk rows with no headline or a
        # zero timestamp — drop them before ranking so they can't be
        # mistaken for the most recent article.
        if not r.get("headline") or not r.get("datetime"):
            continue
        published_at = datetime.fromtimestamp(r["datetime"])
        articles.append({
            "title": _clean_text(r["headline"]),
            "source": r.get("source", "unknown"),
            "url": r.get("url", ""),
            "summary": _clean_text(r.get("summary", "")),
            "category": r.get("category", "general"),
            "published_at": published_at,
            "relevance": _relevance_from_recency(published_at, window_start, window_end),
            "external_id": r.get("id"),
        })

    articles.sort(key=lambda a: a["published_at"], reverse=True)
    return articles[:limit]
