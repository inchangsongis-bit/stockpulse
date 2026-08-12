from datetime import datetime, timedelta

import pytest

from models import NewsArticle


async def seed_articles(session_factory, ticker="SPY", n=3, start_days_ago=2):
    async with session_factory() as session:
        base = datetime.now() - timedelta(days=start_days_ago)
        for i in range(n):
            session.add(NewsArticle(
                ticker=ticker,
                title=f"Article {i}",
                source="Reuters",
                url=f"https://example.com/article-{i}",
                summary="Summary",
                category="company",
                published_at=base + timedelta(hours=i),
                relevance=0.8,
                external_id=i,
                sentiment=0.5,
                source_credibility=0.9,
                expected_impact="medium",
                reasoning="reasoning",
                sentiment_scored_at=datetime.now(),
            ))
        await session.commit()


@pytest.mark.asyncio
async def test_news_returns_persisted_articles_newest_first(client, session_factory):
    await seed_articles(session_factory, ticker="SPY", n=3)

    resp = await client.get("/api/stocks/SPY/news")
    assert resp.status_code == 200
    body = resp.json()

    assert body["ticker"] == "SPY"
    assert body["count"] == 3
    published_dates = [a["published_at"] for a in body["articles"]]
    assert published_dates == sorted(published_dates, reverse=True)
    assert body["articles"][0]["sentiment"] == 0.5


@pytest.mark.asyncio
async def test_news_scoped_to_ticker_and_case_insensitive(client, session_factory):
    await seed_articles(session_factory, ticker="SPY", n=2)
    await seed_articles(session_factory, ticker="AAPL", n=1)

    resp = await client.get("/api/stocks/spy/news")
    assert resp.status_code == 200
    assert resp.json()["count"] == 2


@pytest.mark.asyncio
async def test_news_returns_empty_list_not_404_for_unseeded_ticker(client):
    resp = await client.get("/api/stocks/NOPE/news")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["articles"] == []


@pytest.mark.asyncio
async def test_news_respects_days_window(client, session_factory):
    await seed_articles(session_factory, ticker="SPY", n=3, start_days_ago=2)

    resp = await client.get("/api/stocks/SPY/news?days=1")
    assert resp.status_code == 200
    # articles published ~2 days ago should fall outside a 1-day window
    assert resp.json()["count"] < 3
