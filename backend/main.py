"""
StockPulse API — FastAPI backend
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db, async_session
from models import OHLCV, Watchlist
from mock_data import generate_daily_ohlcv
from routers import stocks, signals, watchlist, pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: create tables and seed mock data if DB is empty."""
    await init_db()
    await seed_mock_data()
    yield


app = FastAPI(
    title="StockPulse API",
    description="AI-powered stock analysis pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(stocks.router)
app.include_router(signals.router)
app.include_router(watchlist.router)
app.include_router(pipeline.router)


@app.get("/")
async def root():
    return {
        "app": "StockPulse",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": [
            "GET  /api/stocks/{ticker}/ohlcv",
            "GET  /api/stocks/{ticker}/latest",
            "GET  /api/signals/{ticker}",
            "POST /api/pipeline/run/{ticker}",
            "GET  /api/watchlist/",
            "POST /api/watchlist/",
        ],
    }


async def seed_mock_data():
    """Seed the DB with mock SPY data if it's empty."""
    async with async_session() as db:
        # Check if data already exists
        from sqlalchemy import select, func
        count_result = await db.execute(
            select(func.count()).select_from(OHLCV).where(OHLCV.ticker == "SPY")
        )
        count = count_result.scalar()

        if count and count > 0:
            print(f"Database already has {count} SPY bars, skipping seed.")
            return

        print("Seeding mock SPY data...")

        # Generate 3 years of daily bars
        bars = generate_daily_ohlcv(ticker="SPY")
        print(f"  Generated {len(bars)} daily bars")

        for bar in bars:
            db.add(OHLCV(**bar))

        # Add SPY to watchlist
        db.add(Watchlist(ticker="SPY"))

        await db.commit()
        print(f"  Seeded {len(bars)} OHLCV bars + SPY watchlist entry")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
