"""
Polygon.io market data client.
Docs: https://polygon.io/docs/rest/stocks/aggregates/custom-bars
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import httpx

from config import get_settings


class PolygonError(Exception):
    pass


async def fetch_daily_ohlcv(ticker: str, days: int = 730) -> List[Dict[str, Any]]:
    """Fetch daily OHLCV bars for a ticker from Polygon's aggregates endpoint."""
    api_key = get_settings().polygon_api_key
    if not api_key:
        raise PolygonError("POLYGON_API_KEY is not configured")

    end = datetime.now().date()
    start = end - timedelta(days=days)

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": api_key}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params)

    if resp.status_code != 200:
        raise PolygonError(f"Polygon API error {resp.status_code}: {resp.text[:200]}")

    body = resp.json()
    if body.get("status") not in ("OK", "DELAYED"):
        raise PolygonError(f"Polygon API returned status={body.get('status')}: {body}")

    bars = []
    for r in body.get("results", []):
        bars.append({
            "ticker": ticker.upper(),
            "timestamp": datetime.fromtimestamp(r["t"] / 1000),
            "open": r["o"],
            "high": r["h"],
            "low": r["l"],
            "close": r["c"],
            "volume": int(r["v"]),
            "vwap": r.get("vw"),
            "num_trades": r.get("n"),
        })
    return bars
