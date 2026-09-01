"""
Forecast router — the short-term (5-minute-ahead) direction prediction,
separate from the pipeline's BUY/SELL/HOLD signal (which has a 2-4 week
horizon and is a different kind of call entirely). See analysis/forecast.py.
"""

import json

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from analysis.forecast import ForecastUnavailable, predict_direction
from database import get_db
from models import OHLCV, Signal

router = APIRouter(prefix="/api/forecast", tags=["forecast"])

# The longest rolling feature window is 20 bars; pull a bit more so the
# most recent row is guaranteed fully warmed up.
_BARS_NEEDED = 60
_MIN_BARS = 25


@router.get("/{ticker}")
async def get_forecast(ticker: str, db: AsyncSession = Depends(get_db)):
    """
    Predict whether a ticker's price is more likely to be up or down 5
    minutes from now, from recent minute-bar technical/volume patterns
    plus a small nudge from its current news sentiment.
    """
    ticker = ticker.upper()

    result = await db.execute(
        select(OHLCV)
        .where(OHLCV.ticker == ticker, OHLCV.interval == "minute")
        .order_by(desc(OHLCV.timestamp))
        .limit(_BARS_NEEDED)
    )
    rows = list(reversed(result.scalars().all()))
    if len(rows) < _MIN_BARS:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough recent minute data for {ticker} — sync minute data first.",
        )

    bars = pd.DataFrame([
        {
            "timestamp": r.timestamp,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
        }
        for r in rows
    ])

    sig_result = await db.execute(
        select(Signal).where(Signal.ticker == ticker).order_by(desc(Signal.timestamp)).limit(1)
    )
    sig_row = sig_result.scalar_one_or_none()
    sentiment = 0.0
    if sig_row and sig_row.factors_json:
        try:
            sentiment = json.loads(sig_row.factors_json).get("sentiment", {}).get("score", 0.0)
        except (json.JSONDecodeError, AttributeError):
            pass

    try:
        prediction = predict_direction(bars, sentiment=sentiment)
    except ForecastUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {
        "ticker": ticker,
        "as_of": rows[-1].timestamp.isoformat(),
        **prediction,
    }
