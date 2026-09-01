"""
Runs the analysis pipeline for every watchlist ticker, one at a time.
Shared by the manual "Refresh All" endpoint (routers/pipeline.py) and the
daily scheduled job (main.py) so both drive the same run, with one
in-memory progress/status they can both be polled through.

Sequential by design: each ticker run already takes ~15-30s (mostly Claude
latency for sentiment/strategy reasoning), which naturally paces well
within provider rate limits — no extra throttling needed.
"""

from datetime import datetime

from sqlalchemy import select

from database import async_session
from models import Watchlist
from services.pipeline_runner import run_pipeline_for_ticker

_status = {
    "running": False,
    "trigger": None,  # "manual" | "scheduled"
    "total": 0,
    "completed": 0,
    "current_ticker": None,
    "started_at": None,
    "finished_at": None,
    "errors": {},
}


def get_status() -> dict:
    return dict(_status, errors=dict(_status["errors"]))


async def run_all(trigger: str = "manual") -> None:
    """Run the pipeline for every watchlist ticker. No-ops if a run (manual
    or scheduled) is already in progress, so the two triggers can't overlap."""
    if _status["running"]:
        return

    async with async_session() as db:
        result = await db.execute(select(Watchlist.ticker))
        tickers = [row[0] for row in result.all()]

    _status.update(
        running=True,
        trigger=trigger,
        total=len(tickers),
        completed=0,
        current_ticker=None,
        started_at=datetime.now().isoformat(),
        finished_at=None,
        errors={},
    )

    try:
        for ticker in tickers:
            _status["current_ticker"] = ticker
            try:
                async with async_session() as db:
                    await run_pipeline_for_ticker(ticker, db)
            except Exception as e:
                _status["errors"][ticker] = str(e)
            _status["completed"] += 1
    finally:
        _status["current_ticker"] = None
        _status["running"] = False
        _status["finished_at"] = datetime.now().isoformat()
