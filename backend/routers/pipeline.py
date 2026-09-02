"""
Pipeline router — triggers the full agent pipeline for a ticker, or for
the whole watchlist at once.
"""

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from services.bulk_pipeline import get_status, run_all
from services.pipeline_runner import run_pipeline_for_ticker

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.post("/run/{ticker}")
async def run_pipeline(ticker: str, db: AsyncSession = Depends(get_db)):
    """Run the full analysis pipeline for a ticker."""
    return await run_pipeline_for_ticker(ticker, db)


@router.post("/run-all")
async def run_pipeline_for_all(background_tasks: BackgroundTasks):
    """
    Kick off a background run of the full pipeline for every watchlist
    ticker (the same "Refresh All" this session ran by hand as a bash loop
    before this endpoint existed). Returns immediately — poll
    GET /run-all/status for progress. No-ops (with status "already_running")
    if a run, manual or scheduled, is already in flight.
    """
    status = get_status()
    if status["running"]:
        return {"status": "already_running", **status}
    background_tasks.add_task(run_all, trigger="manual")
    return {"status": "started"}


@router.get("/run-all/status")
async def get_pipeline_run_all_status():
    """Progress of the current or most recent 'Refresh All' run (manual or
    scheduled)."""
    return get_status()
