from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from models import NewsArticle


async def seed_article(session_factory, ticker="SPY", url_suffix="0", **overrides):
    defaults = dict(
        ticker=ticker,
        title="Some headline",
        source="Reuters",
        url=f"https://example.com/{ticker.lower()}-{url_suffix}",
        summary="Some summary",
        category="company",
        published_at=datetime.now() - timedelta(hours=1),
        relevance=0.9,
    )
    defaults.update(overrides)
    async with session_factory() as session:
        session.add(NewsArticle(**defaults))
        await session.commit()


@pytest.mark.asyncio
async def test_score_news_sentiment_scores_only_unscored_articles(client, session_factory, monkeypatch):
    await seed_article(session_factory, url_suffix="unscored")
    await seed_article(
        session_factory,
        url_suffix="already-scored",
        sentiment=0.6,
        source_credibility=0.95,
        expected_impact="high",
        reasoning="Previously scored by Claude",
        sentiment_scored_at=datetime.now(),
    )

    monkeypatch.setattr("routers.stocks.score_text", lambda text: (0.8, "positive", 0.9))

    resp = await client.post("/api/stocks/SPY/news/score")
    assert resp.status_code == 200
    assert resp.json() == {"ticker": "SPY", "scored": 1}

    async with session_factory() as session:
        result = await session.execute(select(NewsArticle).where(NewsArticle.ticker == "SPY"))
        rows = {row.url: row for row in result.scalars().all()}

    unscored = rows["https://example.com/spy-unscored"]
    assert unscored.sentiment == 0.8
    assert unscored.source_credibility == 0.95  # Reuters
    assert unscored.expected_impact == "high"  # relevance 0.9 > 0.85
    assert "FinBERT" in unscored.reasoning
    assert unscored.sentiment_scored_at is not None

    already_scored = rows["https://example.com/spy-already-scored"]
    assert already_scored.sentiment == 0.6  # untouched
    assert already_scored.reasoning == "Previously scored by Claude"


@pytest.mark.asyncio
async def test_score_news_sentiment_maps_negative_and_neutral_labels(client, session_factory, monkeypatch):
    await seed_article(session_factory, url_suffix="neg")

    calls = []

    def fake_score(text):
        calls.append(text)
        return (-0.7, "negative", 0.7)

    monkeypatch.setattr("routers.stocks.score_text", fake_score)

    resp = await client.post("/api/stocks/SPY/news/score")
    assert resp.json() == {"ticker": "SPY", "scored": 1}
    assert len(calls) == 1
    assert "Some headline" in calls[0]

    async with session_factory() as session:
        result = await session.execute(select(NewsArticle).where(NewsArticle.ticker == "SPY"))
        row = result.scalars().one()
    assert row.sentiment == -0.7
    assert "negative" in row.reasoning


@pytest.mark.asyncio
async def test_score_news_sentiment_no_unscored_articles_is_a_noop(client, monkeypatch):
    called = False

    def fake_score(text):
        nonlocal called
        called = True
        return (0.0, "neutral", 0.5)

    monkeypatch.setattr("routers.stocks.score_text", fake_score)

    resp = await client.post("/api/stocks/NOPE/news/score")
    assert resp.status_code == 200
    assert resp.json() == {"ticker": "NOPE", "scored": 0}
    assert called is False


@pytest.mark.asyncio
async def test_score_news_sentiment_respects_limit(client, session_factory, monkeypatch):
    for i in range(3):
        await seed_article(session_factory, url_suffix=str(i))

    monkeypatch.setattr("routers.stocks.score_text", lambda text: (0.5, "positive", 0.5))

    resp = await client.post("/api/stocks/SPY/news/score?limit=2")
    assert resp.json() == {"ticker": "SPY", "scored": 2}
