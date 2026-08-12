import pytest

from agents.news_researcher import NewsResearcherAgent
from data_sources.finnhub import FinnhubError


@pytest.mark.asyncio
async def test_mock_mode_returns_mock_articles():
    agent = NewsResearcherAgent()
    research = await agent.run("SPY", {"use_mock": True})

    assert research["ticker"] == "SPY"
    assert research["article_count"] == len(research["articles"])
    assert research["article_count"] > 0
    for a in research["articles"]:
        assert set(a.keys()) == {
            "title", "source", "url", "summary", "category", "published_at", "relevance", "external_id",
        }
        assert a["external_id"] is None  # mock articles have no Finnhub id


@pytest.mark.asyncio
async def test_live_mode_passes_through_external_id(monkeypatch):
    from datetime import datetime

    async def fake_fetch(ticker, **kwargs):
        return [{
            "title": "Real headline",
            "source": "Reuters",
            "url": "https://example.com/real",
            "summary": "Real summary",
            "category": "company",
            "published_at": datetime.now(),
            "relevance": 0.9,
            "external_id": 42,
        }]

    monkeypatch.setattr("data_sources.finnhub.fetch_company_news", fake_fetch)

    agent = NewsResearcherAgent()
    research = await agent.run("SPY", {"use_mock": False})

    assert research["article_count"] == 1
    assert research["articles"][0]["external_id"] == 42
    assert research["articles"][0]["title"] == "Real headline"


@pytest.mark.asyncio
async def test_live_mode_finnhub_error_returns_empty_article_list(monkeypatch):
    async def fake_fetch(ticker, **kwargs):
        raise FinnhubError("boom")

    monkeypatch.setattr("data_sources.finnhub.fetch_company_news", fake_fetch)

    agent = NewsResearcherAgent()
    research = await agent.run("SPY", {"use_mock": False})

    assert research["article_count"] == 0
    assert research["articles"] == []
