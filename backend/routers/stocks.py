from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, delete
from datetime import datetime, timedelta
from typing import Optional
from database import get_db
from models import OHLCV, NewsArticle
from data_sources.polygon import fetch_ohlcv, PolygonError
from data_sources.finnhub import fetch_company_news, FinnhubError

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
        .where(
            OHLCV.ticker == ticker.upper(),
            OHLCV.interval == interval,
            OHLCV.timestamp >= since,
        )
        .order_by(OHLCV.timestamp.asc())
    )
    rows = result.scalars().all()
    return {
        "ticker": ticker.upper(),
        "interval": interval,
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


# Minute-level history gets large fast (Polygon caps a single request at
# 50,000 bars — a few weeks of 1-min SPY bars), so it gets a much smaller
# default/max window than daily bars.
_SYNC_LIMITS = {
    "daily": {"default_days": 730, "max_days": 1825},
    "minute": {"default_days": 5, "max_days": 30},
}


@router.post("/{ticker}/sync")
async def sync_ohlcv(
    ticker: str,
    interval: str = Query(default="daily", regex="^(daily|minute)$"),
    days: Optional[int] = Query(default=None, ge=1),
    db: AsyncSession = Depends(get_db),
):
    """Replace stored OHLCV history for a ticker with real data from Polygon."""
    ticker = ticker.upper()
    limits = _SYNC_LIMITS[interval]
    days = days if days is not None else limits["default_days"]
    if days > limits["max_days"]:
        raise HTTPException(
            status_code=400,
            detail=f"days must be <= {limits['max_days']} for interval={interval}",
        )

    try:
        bars = await fetch_ohlcv(ticker, interval=interval, days=days)
    except PolygonError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if not bars:
        raise HTTPException(status_code=502, detail=f"Polygon returned no data for {ticker}")

    await db.execute(delete(OHLCV).where(OHLCV.ticker == ticker, OHLCV.interval == interval))
    for bar in bars:
        db.add(OHLCV(**bar))
    await db.commit()

    return {
        "ticker": ticker,
        "interval": interval,
        "synced_bars": len(bars),
        "range": {
            "start": bars[0]["timestamp"].isoformat(),
            "end": bars[-1]["timestamp"].isoformat(),
        },
        "latest_close": bars[-1]["close"],
    }


@router.post("/{ticker}/news/sync")
async def sync_news(
    ticker: str,
    days: int = Query(default=7, ge=1, le=30),
    limit: int = Query(default=15, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch and persist real news articles for a ticker WITHOUT scoring
    sentiment — no Claude calls, just retrieval + storage. Deliberately
    separate from the analysis pipeline (POST /api/pipeline/run/{ticker}),
    which is the only other place news gets fetched, but couples it to a
    full (paid) sentiment-scoring pass. Existing sentiment columns on
    already-persisted articles are left untouched; new rows are stored
    with those columns null until something scores them later.
    """
    ticker = ticker.upper()
    try:
        articles = await fetch_company_news(ticker, days=days, limit=limit)
    except FinnhubError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if not articles:
        raise HTTPException(status_code=502, detail=f"No news found for {ticker}")

    urls = [a["url"] for a in articles if a.get("url")]
    existing_by_url = {}
    if urls:
        existing_result = await db.execute(
            select(NewsArticle).where(NewsArticle.ticker == ticker, NewsArticle.url.in_(urls))
        )
        existing_by_url = {row.url: row for row in existing_result.scalars().all()}

    new_count = 0
    for a in articles:
        row = existing_by_url.get(a.get("url"))
        if row is None:
            row = NewsArticle(ticker=ticker, url=a.get("url"))
            db.add(row)
            new_count += 1
        row.title = a["title"]
        row.source = a["source"]
        row.summary = a["summary"]
        row.category = a["category"]
        row.published_at = a["published_at"]
        row.relevance = a["relevance"]
        row.external_id = a.get("external_id")

    await db.commit()

    return {
        "ticker": ticker,
        "fetched": len(articles),
        "new": new_count,
        "updated": len(articles) - new_count,
    }


@router.get("/{ticker}/news")
async def get_news(
    ticker: str,
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """Get persisted news articles + cached sentiment for a ticker."""
    since = datetime.now() - timedelta(days=days)
    result = await db.execute(
        select(NewsArticle)
        .where(NewsArticle.ticker == ticker.upper(), NewsArticle.published_at >= since)
        .order_by(desc(NewsArticle.published_at))
    )
    rows = result.scalars().all()
    return {
        "ticker": ticker.upper(),
        "count": len(rows),
        "articles": [
            {
                "title": r.title,
                "source": r.source,
                "url": r.url,
                "summary": r.summary,
                "category": r.category,
                "published_at": r.published_at.isoformat() if r.published_at else None,
                "relevance": r.relevance,
                "sentiment": r.sentiment,
                "source_credibility": r.source_credibility,
                "expected_impact": r.expected_impact,
                "reasoning": r.reasoning,
            }
            for r in rows
        ],
    }


@router.get("/{ticker}/latest")
async def get_latest(ticker: str, db: AsyncSession = Depends(get_db)):
    """Get the most recent OHLCV bar, whichever interval it came from."""
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
        "interval": row.interval,
        "timestamp": row.timestamp.isoformat(),
        "open": row.open,
        "high": row.high,
        "low": row.low,
        "close": row.close,
        "volume": row.volume,
    }
