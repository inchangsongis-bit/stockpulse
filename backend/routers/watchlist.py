from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from pydantic import BaseModel
from database import get_db
from models import Watchlist

router = APIRouter(prefix="/api/watchlist", tags=["watchlist"])


class AddTickerRequest(BaseModel):
    ticker: str


@router.get("/")
async def list_watchlist(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Watchlist).order_by(Watchlist.added_at.desc()))
    rows = result.scalars().all()
    return {
        "tickers": [
            {"ticker": r.ticker, "added_at": r.added_at.isoformat() if r.added_at else None}
            for r in rows
        ]
    }


@router.post("/")
async def add_ticker(req: AddTickerRequest, db: AsyncSession = Depends(get_db)):
    ticker = req.ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker cannot be empty")

    # Check if already exists
    existing = await db.execute(select(Watchlist).where(Watchlist.ticker == ticker))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"{ticker} already in watchlist")

    entry = Watchlist(ticker=ticker)
    db.add(entry)
    await db.commit()
    return {"status": "added", "ticker": ticker}


@router.delete("/{ticker}")
async def remove_ticker(ticker: str, db: AsyncSession = Depends(get_db)):
    ticker = ticker.upper()
    result = await db.execute(delete(Watchlist).where(Watchlist.ticker == ticker))
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail=f"{ticker} not found in watchlist")
    return {"status": "removed", "ticker": ticker}
