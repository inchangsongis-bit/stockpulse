"""
Runs the full analysis pipeline for a single ticker and persists the
result. Pulled out of routers/pipeline.py so it can be called both from
the single-ticker HTTP endpoint and from services/bulk_pipeline.py's
run-every-watchlist-ticker loop, without those two importing each other.
"""

import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.orchestrator import PipelineOrchestrator
from config import get_settings
from models import OHLCV, NewsArticle, Signal


async def run_pipeline_for_ticker(ticker: str, db: AsyncSession) -> dict:
    """Run the full analysis pipeline for a ticker, persisting the signal
    and (for live runs) fetched news + sentiment."""
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

    # Use real news (Finnhub) when a key is configured, mock data otherwise.
    use_mock = not bool(get_settings().finnhub_api_key)

    # For live runs, avoid re-scoring articles we've already persisted a
    # sentiment score for — the sentiment agent will skip Claude calls for
    # any article whose URL is in this cache.
    cached_sentiment = {}
    if not use_mock:
        cached_result = await db.execute(
            select(NewsArticle).where(NewsArticle.ticker == ticker, NewsArticle.sentiment.isnot(None))
        )
        cached_sentiment = {
            row.url: {
                "title": row.title,
                "source": row.source,
                "url": row.url,
                "sentiment": row.sentiment,
                "source_credibility": row.source_credibility,
                "expected_impact": row.expected_impact,
                "reasoning": row.reasoning,
            }
            for row in cached_result.scalars().all()
        }

    # Run pipeline
    orchestrator = PipelineOrchestrator()
    pipeline_result = await orchestrator.run_pipeline(
        ticker=ticker,
        ohlcv_data=ohlcv_data,
        use_mock=use_mock,
        cached_sentiment=cached_sentiment,
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

    # Persist real news + sentiment so future runs can reuse it (see
    # cached_sentiment above). Mock runs stay fully ephemeral, matching
    # today's behavior — no point accumulating fake "reuters.com/mock/..."
    # rows.
    if not use_mock:
        articles = pipeline_result["research"]["articles"]
        scores_by_title = {s["title"]: s for s in pipeline_result["sentiment_profile"]["article_scores"]}
        urls = [a["url"] for a in articles if a.get("url")]
        existing_by_url = {}
        if urls:
            existing_result = await db.execute(
                select(NewsArticle).where(NewsArticle.ticker == ticker, NewsArticle.url.in_(urls))
            )
            existing_by_url = {row.url: row for row in existing_result.scalars().all()}

        for a in articles:
            row = existing_by_url.get(a.get("url"))
            if row is None:
                row = NewsArticle(ticker=ticker, url=a.get("url"))
                db.add(row)
            row.title = a["title"]
            row.source = a["source"]
            row.summary = a["summary"]
            row.category = a["category"]
            # news_researcher.py serializes published_at to an ISO string
            # for the HTTP response; parse it back for the DateTime column.
            published_at = a["published_at"]
            row.published_at = (
                datetime.fromisoformat(published_at) if isinstance(published_at, str) else published_at
            )
            row.relevance = a["relevance"]
            row.external_id = a.get("external_id")
            score = scores_by_title.get(a["title"])
            if score:
                row.sentiment = score["sentiment"]
                row.source_credibility = score["source_credibility"]
                row.expected_impact = score["expected_impact"]
                row.reasoning = score["reasoning"]
                row.sentiment_scored_at = datetime.now()

        await db.commit()

    return pipeline_result
