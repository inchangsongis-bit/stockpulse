"""
Syncs OHLCV history for a single ticker from Polygon. Pulled out of
routers/stocks.py so both the HTTP endpoint and services/daily_digest.py's
pre-market bulk refresh share one implementation.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from data_sources.polygon import PolygonError, fetch_ohlcv
from models import OHLCV

# Minute-level history gets large fast (Polygon caps a single request at
# 50,000 bars — a few weeks of 1-min SPY bars), so it gets a much smaller
# default/max window than daily bars.
SYNC_LIMITS = {
    "daily": {"default_days": 730, "max_days": 1825},
    "minute": {"default_days": 5, "max_days": 30},
}

# When a ticker+interval already has stored bars, sync only fetches from
# the last stored bar forward instead of the full window — but with this
# much overlap, to pick up any late-arriving/corrected bars Polygon may
# still be revising for the most recent trading session(s).
SYNC_OVERLAP_DAYS = {"daily": 5, "minute": 1}


class SyncValidationError(Exception):
    pass


async def sync_ticker_ohlcv(ticker: str, interval: str, days: Optional[int], db: AsyncSession) -> dict:
    """
    First sync for a ticker+interval fetches and stores the full `days`
    window (or the default). Every sync after that is incremental: it
    fetches only from the last stored bar forward (plus a small overlap
    to catch corrections to recent bars) and replaces just that
    overlapping tail — `days` is ignored once a ticker has history, so a
    caller passing a short window can't wipe out years of previously-
    synced data.

    Raises SyncValidationError for a bad `days` value, PolygonError for
    any upstream fetch problem — callers translate those into whatever's
    appropriate for their context (HTTP errors for the router, a skip for
    the daily digest's best-effort bulk refresh).
    """
    ticker = ticker.upper()
    limits = SYNC_LIMITS[interval]

    latest_result = await db.execute(
        select(func.max(OHLCV.timestamp)).where(OHLCV.ticker == ticker, OHLCV.interval == interval)
    )
    latest_ts = latest_result.scalar_one_or_none()

    if latest_ts is None:
        fetch_days = days if days is not None else limits["default_days"]
        if fetch_days > limits["max_days"]:
            raise SyncValidationError(f"days must be <= {limits['max_days']} for interval={interval}")
    else:
        overlap = SYNC_OVERLAP_DAYS[interval]
        since_days = (datetime.now() - latest_ts).days + overlap
        fetch_days = max(overlap, min(since_days, limits["max_days"]))

    bars = await fetch_ohlcv(ticker, interval=interval, days=fetch_days)

    if not bars:
        raise PolygonError(f"Polygon returned no data for {ticker}")

    if latest_ts is None:
        await db.execute(delete(OHLCV).where(OHLCV.ticker == ticker, OHLCV.interval == interval))
    else:
        # Only clear the overlapping tail we just re-fetched — older,
        # untouched history is never wiped by a sync.
        await db.execute(
            delete(OHLCV).where(
                OHLCV.ticker == ticker,
                OHLCV.interval == interval,
                OHLCV.timestamp >= bars[0]["timestamp"],
            )
        )
    for bar in bars:
        db.add(OHLCV(**bar))
    await db.commit()

    return {
        "ticker": ticker,
        "interval": interval,
        "mode": "full" if latest_ts is None else "incremental",
        "synced_bars": len(bars),
        "range": {
            "start": bars[0]["timestamp"].isoformat(),
            "end": bars[-1]["timestamp"].isoformat(),
        },
        "latest_close": bars[-1]["close"],
    }
