"""
Pipeline router — triggers the full agent pipeline for a ticker.
"""

import json
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db
from models import OHLCV, Signal
from agents.orchestrator import PipelineOrchestrator

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.post("/run/{ticker}")
async def run_pipeline(ticker: str, db: AsyncSession = Depends(get_db)):
    """Run the full analysis pipeline for a ticker."""
    ticker = ticker.upper()

    # Fetch daily OHLCV data from DB — indicators (SMA/RSI/etc.) assume
    # one bar per day, so minute bars must be excluded here.
    result = await db.execute(
        select(OHLCV)
        .where(OHLCV.ticker == ticker, OHLCV.interval == "daily")
        .order_by(OHLCV.timestamp.asc())
    )
    rows = result.scalars().all()

    ohlcv_data = [
        {
            "timestamp": r.timestamp,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
            "vwap": r.vwap,
        }
        for r in rows
    ]

    if not ohlcv_data:
        return {"error": f"No OHLCV data for {ticker}. Seed the database first."}

    # Run pipeline
    orchestrator = PipelineOrchestrator()
    pipeline_result = await orchestrator.run_pipeline(
        ticker=ticker,
        ohlcv_data=ohlcv_data,
        use_mock=True,
    )

    # Save signal to DB
    sig = pipeline_result["signal"]
    db_signal = Signal(
        ticker=ticker,
        timestamp=datetime.now(),
        action=sig["action"],
        confidence=sig["confidence"],
        reasoning=sig["reasoning"],
        entry_low=sig["entry_low"],
        entry_high=sig["entry_high"],
        target=sig["target"],
        stop_loss=sig["stop_loss"],
        time_horizon=sig["time_horizon"],
        risk_level=sig["risk_level"],
        factors_json=json.dumps(sig["factors"]),
    )
    db.add(db_signal)
    await db.commit()

    return pipeline_result
