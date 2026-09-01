"""
The full pre-market daily job: incrementally sync daily OHLCV for every
watchlist ticker, run the pipeline for all of them (this also fetches
fresh news — no separate news-sync step needed, see routers/pipeline.py),
then email the resulting BUY/SELL/HOLD digest to every active subscriber.

Wired into the scheduler in main.py at 6:25am US/Pacific (5 minutes
before the US market opens) — see main.py's CronTrigger. This replaces
the old close-of-day-only refresh with one comprehensive daily run
instead of two separate bulk-Claude-cost runs per day.
"""

from datetime import datetime

from sqlalchemy import select

from data_sources.polygon import PolygonError
from database import async_session
from models import Subscriber, Watchlist
from services.bulk_pipeline import run_all
from services.email_sender import EmailError, send_email
from services.email_templates import render_daily_digest_html
from services.ohlcv_sync import SyncValidationError, sync_ticker_ohlcv
from services.watchlist_summary import get_summary_rows


async def _sync_all_daily_ohlcv() -> None:
    async with async_session() as db:
        result = await db.execute(select(Watchlist.ticker))
        tickers = [row[0] for row in result.all()]

    for ticker in tickers:
        try:
            async with async_session() as db:
                await sync_ticker_ohlcv(ticker, interval="daily", days=None, db=db)
        except (PolygonError, SyncValidationError):
            # Best-effort — one stale/unreachable ticker shouldn't block
            # the rest of the watchlist from refreshing.
            continue


async def send_daily_digest() -> dict:
    """Returns a small summary dict for logging/testing — not exposed over HTTP."""
    await _sync_all_daily_ohlcv()
    await run_all(trigger="scheduled")

    async with async_session() as db:
        rows = await get_summary_rows(db)
        sub_result = await db.execute(select(Subscriber).where(Subscriber.is_active.is_(True)))
        active_subscribers = sub_result.scalars().all()

    sent, failed = 0, 0
    for sub in active_subscribers:
        html = render_daily_digest_html(rows, as_of=datetime.now(), unsubscribe_token=sub.unsubscribe_token)
        try:
            await send_email(sub.email, "StockPulse Daily Signals", html)
            sent += 1
        except EmailError:
            failed += 1  # best-effort per-recipient — one bad address shouldn't block the rest

    return {"tickers": len(rows), "subscribers": len(active_subscribers), "sent": sent, "failed": failed}
