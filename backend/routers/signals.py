from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from database import get_db
from models import Signal

router = APIRouter(prefix="/api/signals", tags=["signals"])


@router.get("/{ticker}")
async def get_signals(
    ticker: str,
    limit: int = Query(default=10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Get recent signals for a ticker."""
    result = await db.execute(
        select(Signal)
        .where(Signal.ticker == ticker.upper())
        .order_by(desc(Signal.timestamp))
        .limit(limit)
    )
    rows = result.scalars().all()
    return {
        "ticker": ticker.upper(),
        "count": len(rows),
        "signals": [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "action": r.action,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
                "entry_low": r.entry_low,
                "entry_high": r.entry_high,
                "target": r.target,
                "stop_loss": r.stop_loss,
                "time_horizon": r.time_horizon,
                "risk_level": r.risk_level,
            }
            for r in rows
        ],
    }


@router.get("/latest/{ticker}")
async def get_latest_signal(ticker: str, db: AsyncSession = Depends(get_db)):
    """Get the most recent signal for a ticker."""
    result = await db.execute(
        select(Signal)
        .where(Signal.ticker == ticker.upper())
        .order_by(desc(Signal.timestamp))
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if not row:
        return {"error": "No signals found"}
    return {
        "id": row.id,
        "ticker": row.ticker,
        "timestamp": row.timestamp.isoformat(),
        "action": row.action,
        "confidence": row.confidence,
        "reasoning": row.reasoning,
        "entry_low": row.entry_low,
        "entry_high": row.entry_high,
        "target": row.target,
        "stop_loss": row.stop_loss,
        "time_horizon": row.time_horizon,
        "risk_level": row.risk_level,
        "factors_json": row.factors_json,
    }
