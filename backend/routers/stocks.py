from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from datetime import datetime, timedelta
from typing import Optional
from database import get_db
from models import OHLCV

router = APIRouter(prefix="/api/stocks", tags=["stocks"])


@router.get("/{ticker}/ohlcv")
async def get_ohlcv(
    ticker: str,
    days: int = Query(default=90, ge=1, le=1100),
    interval: str = Query(default="daily", regex="^(daily|minute)$"),
    db: AsyncSession = Depends(get_db),
):
    """Get OHLCV data for a ticker."""
    since = datetime.now() - timedelta(days=days)
    result = await db.execute(
        select(OHLCV)
        .where(OHLCV.ticker == ticker.upper(), OHLCV.timestamp >= since)
        .order_by(OHLCV.timestamp.asc())
    )
    rows = result.scalars().all()
    return {
        "ticker": ticker.upper(),
        "count": len(rows),
        "data": [
            {
                "timestamp": r.timestamp.isoformat(),
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "vwap": r.vwap,
            }
            for r in rows
        ],
    }


@router.get("/{ticker}/latest")
async def get_latest(ticker: str, db: AsyncSession = Depends(get_db)):
    """Get the most recent OHLCV bar."""
    result = await db.execute(
        select(OHLCV)
        .where(OHLCV.ticker == ticker.upper())
        .order_by(desc(OHLCV.timestamp))
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if not row:
        return {"error": "No data found"}
    return {
        "ticker": row.ticker,
        "timestamp": row.timestamp.isoformat(),
        "open": row.open,
        "high": row.high,
        "low": row.low,
        "close": row.close,
        "volume": row.volume,
    }
