"""
Builds the watchlist-tickers-with-latest-price-and-signal rows used by
GET /api/watchlist/summary and, identically, by the daily digest email
(services/daily_digest.py) — pulled out so both share one query.
"""

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from models import OHLCV, Signal, Watchlist


async def get_summary_rows(db: AsyncSession) -> list:
    wl_result = await db.execute(select(Watchlist).order_by(Watchlist.added_at.desc()))
    tickers = [r.ticker for r in wl_result.scalars().all()]
    if not tickers:
        return []

    # Latest price per ticker, any interval — most recent bar wins.
    #
    # One indexed lookup per ticker rather than a single clever query.
    # The obvious approaches are both far slower here: streaming every
    # bar back and taking the first per ticker in Python made this a
    # 34-second endpoint once the minute-bar backfill pushed ohlcv past
    # 13M rows, and a ROW_NUMBER() window function still made SQLite sort
    # each ticker's rows from scratch, because its planner won't use the
    # (ticker, timestamp) index to satisfy a window ORDER BY.
    #
    # A plain per-ticker "ORDER BY timestamp DESC LIMIT 1" does use that
    # index — 51 of them measure at ~3ms in total, against 5.3s for the
    # window-function version.
    latest_price: dict = {}
    for ticker in tickers:
        row = await db.execute(
            select(OHLCV.close)
            .where(OHLCV.ticker == ticker)
            .order_by(desc(OHLCV.timestamp))
            .limit(1)
        )
        price = row.scalar_one_or_none()
        if price is not None:
            latest_price[ticker] = price

    # Latest signal per ticker, same "first seen per ticker wins" approach.
    sig_result = await db.execute(
        select(Signal).where(Signal.ticker.in_(tickers)).order_by(Signal.ticker, desc(Signal.timestamp))
    )
    latest_signal: dict = {}
    for row in sig_result.scalars().all():
        latest_signal.setdefault(row.ticker, row)

    return [
        {
            "ticker": t,
            "price": latest_price.get(t),
            "signal": (
                {
                    "action": latest_signal[t].action,
                    "confidence": latest_signal[t].confidence,
                    "timestamp": latest_signal[t].timestamp.isoformat(),
                }
                if t in latest_signal
                else None
            ),
        }
        for t in tickers
    ]
