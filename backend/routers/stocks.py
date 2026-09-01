from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, delete, func
from datetime import datetime, timedelta
from typing import Optional
from database import get_db
from models import OHLCV, NewsArticle
from data_sources.polygon import fetch_ohlcv, PolygonError
from data_sources.finnhub import fetch_company_news, FinnhubError
from analysis.finbert_sentiment import score_text
from analysis.news_heuristics import source_credibility, impact_from_relevance

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

# When a ticker+interval already has stored bars, sync only fetches from
# the last stored bar forward instead of the full window — but with this
# much overlap, to pick up any late-arriving/corrected bars Polygon may
# still be revising for the most recent trading session(s).
_SYNC_OVERLAP_DAYS = {"daily": 5, "minute": 1}


@router.post("/{ticker}/sync")
async def sync_ohlcv(
    ticker: str,
    interval: str = Query(default="daily", regex="^(daily|minute)$"),
    days: Optional[int] = Query(default=None, ge=1),
    db: AsyncSession = Depends(get_db),
):
    """
    Sync OHLCV history for a ticker from Polygon.

    First sync for a ticker+interval fetches and stores the full `days`
    window (or the default). Every sync after that is incremental: it
    fetches only from the last stored bar forward (plus a small overlap
    to catch corrections to recent bars) and replaces just that
    overlapping tail — `days` is ignored once a ticker has history, so
    syncing while viewing a short range pill (e.g. "1M") can no longer
    wipe out years of previously-synced data.
    """
    ticker = ticker.upper()
    limits = _SYNC_LIMITS[interval]

    latest_result = await db.execute(
        select(func.max(OHLCV.timestamp)).where(OHLCV.ticker == ticker, OHLCV.interval == interval)
    )
    latest_ts = latest_result.scalar_one_or_none()

    if latest_ts is None:
        fetch_days = days if days is not None else limits["default_days"]
        if fetch_days > limits["max_days"]:
            raise HTTPException(
                status_code=400,
                detail=f"days must be <= {limits['max_days']} for interval={interval}",
            )
    else:
        overlap = _SYNC_OVERLAP_DAYS[interval]
        since_days = (datetime.now() - latest_ts).days + overlap
        fetch_days = max(overlap, min(since_days, limits["max_days"]))

    try:
        bars = await fetch_ohlcv(ticker, interval=interval, days=fetch_days)
    except PolygonError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if not bars:
        raise HTTPException(status_code=502, detail=f"Polygon returned no data for {ticker}")

    if latest_ts is None:
        await db.execute(delete(OHLCV).where(OHLCV.ticker == ticker, OHLCV.interval == interval))
    else:
        # Only clear the overlapping tail we just re-fetched — older,
        # untouched history is never wiped by a sync.
        await db.execute(
            delete(OHLCV).where(
                OHLCV.ticker == ticker,
                OHLCV.interval == interval,
                OHLCV.timestamp >= bars[0]["timestamp"],
            )
        )
    for bar in bars:
        db.add(OHLCV(**bar))
    await db.commit()

    return {
        "ticker": ticker,
        "interval": interval,
        "mode": "full" if latest_ts is None else "incremental",
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


@router.post("/{ticker}/news/score")
async def score_news_sentiment(
    ticker: str,
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """
    Score persisted-but-unscored news articles for a ticker via FinBERT —
    a free, local model (no Claude calls, no per-article cost). Only
    fills in articles where sentiment is currently null; never touches
    an article that already has a score (e.g. from a prior Claude-scored
    pipeline run), so this is safe to run repeatedly without downgrading
    better-quality existing scores.
    """
    ticker = ticker.upper()
    result = await db.execute(
        select(NewsArticle)
        .where(NewsArticle.ticker == ticker, NewsArticle.sentiment.is_(None))
        .limit(limit)
    )
    rows = result.scalars().all()

    for row in rows:
        sentiment, label, confidence = score_text(f"{row.title}. {row.summary or ''}")
        row.sentiment = sentiment
        row.source_credibility = source_credibility(row.source or "")
        row.expected_impact = impact_from_relevance(row.relevance or 0.5)
        row.reasoning = f"FinBERT: {label} (confidence {confidence:.2f})"
        row.sentiment_scored_at = datetime.now()

    await db.commit()

    return {"ticker": ticker, "scored": len(rows)}


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
