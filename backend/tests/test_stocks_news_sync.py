from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from data_sources.finnhub import FinnhubError
from models import NewsArticle


def fake_articles(n=3, ticker="SPY"):
    now = datetime.now()
    return [
        {
            "title": f"Article {i}",
            "source": "Reuters",
            "url": f"https://example.com/{ticker.lower()}-article-{i}",
            "summary": f"Summary {i}",
            "category": "company",
            "published_at": now - timedelta(hours=i),
            "relevance": 0.9,
            "external_id": i,
        }
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_sync_news_persists_articles_without_sentiment(client, monkeypatch):
    async def fake_fetch(ticker, days=7, limit=15):
        return fake_articles(3, ticker)

    monkeypatch.setattr("routers.stocks.fetch_company_news", fake_fetch)

    resp = await client.post("/api/stocks/SPY/news/sync")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"ticker": "SPY", "fetched": 3, "new": 3, "updated": 0}

    news_resp = await client.get("/api/stocks/SPY/news")
    news_body = news_resp.json()
    assert news_body["count"] == 3
    for a in news_body["articles"]:
        assert a["sentiment"] is None
        assert a["reasoning"] is None


@pytest.mark.asyncio
async def test_sync_news_updates_content_without_clobbering_existing_sentiment(
    client, session_factory, monkeypatch
):
    async with session_factory() as session:
        session.add(NewsArticle(
            ticker="SPY",
            url="https://example.com/spy-article-0",
            title="Stale title",
            source="Reuters",
            summary="Stale summary",
            category="company",
            published_at=datetime.now() - timedelta(days=1),
            relevance=0.5,
            sentiment=0.8,
            source_credibility=0.9,
            expected_impact="high",
            reasoning="Previously scored by Claude",
            sentiment_scored_at=datetime.now() - timedelta(days=1),
        ))
        await session.commit()

    async def fake_fetch(ticker, days=7, limit=15):
        return fake_articles(1, "SPY")  # same url (index 0), fresh content

    monkeypatch.setattr("routers.stocks.fetch_company_news", fake_fetch)

    resp = await client.post("/api/stocks/SPY/news/sync")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"ticker": "SPY", "fetched": 1, "new": 0, "updated": 1}

    news_resp = await client.get("/api/stocks/SPY/news")
    article = news_resp.json()["articles"][0]
    assert article["title"] == "Article 0"  # content refreshed
    assert article["sentiment"] == 0.8  # existing score preserved, not wiped
    assert article["reasoning"] == "Previously scored by Claude"


@pytest.mark.asyncio
async def test_sync_news_returns_502_on_provider_error(client, monkeypatch):
    async def fake_fetch(ticker, days=7, limit=15):
        raise FinnhubError("boom")

    monkeypatch.setattr("routers.stocks.fetch_company_news", fake_fetch)

    resp = await client.post("/api/stocks/SPY/news/sync")
    assert resp.status_code == 502


@pytest.mark.asyncio
async def test_sync_news_returns_502_when_no_articles_found(client, monkeypatch):
    async def fake_fetch(ticker, days=7, limit=15):
        return []

    monkeypatch.setattr("routers.stocks.fetch_company_news", fake_fetch)

    resp = await client.post("/api/stocks/SPY/news/sync")
    assert resp.status_code == 502
