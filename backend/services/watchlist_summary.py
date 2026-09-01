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

    # Latest price per ticker, any interval — most recent bar wins. Rows
    # come back ordered per-ticker by recency, so the first one seen per
    # ticker is its latest.
    price_result = await db.execute(
        select(OHLCV.ticker, OHLCV.close, OHLCV.timestamp)
        .where(OHLCV.ticker.in_(tickers))
        .order_by(OHLCV.ticker, desc(OHLCV.timestamp))
    )
    latest_price: dict = {}
    for ticker, close, _ts in price_result.all():
        latest_price.setdefault(ticker, close)

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
