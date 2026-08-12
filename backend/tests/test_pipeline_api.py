from datetime import datetime, timedelta
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select

from models import OHLCV, NewsArticle


@dataclass
class FakeSettings:
    anthropic_api_key: str = ""
    finnhub_api_key: str = ""


@pytest.fixture(autouse=True)
def _default_to_mock_news(monkeypatch):
    """Every test gets mock news (deterministic, no real Finnhub call)
    unless it explicitly overrides routers.pipeline.get_settings itself."""
    monkeypatch.setattr("routers.pipeline.get_settings", lambda: FakeSettings(finnhub_api_key=""))


async def seed_bars(session_factory, ticker="SPY", n=60):
    async with session_factory() as session:
        base = datetime.now() - timedelta(days=n)
        price = 100.0
        for i in range(n):
            price += 0.3
            session.add(OHLCV(
                ticker=ticker,
                timestamp=base + timedelta(days=i),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1_000_000,
                vwap=price,
            ))
        await session.commit()


@pytest.mark.asyncio
async def test_run_pipeline_end_to_end_without_api_key(client, session_factory, monkeypatch):
    # No Anthropic key configured — agents should use their rule-based/template fallbacks
    monkeypatch.setattr(
        "agents.sentiment_analyst.get_settings", lambda: FakeSettings(anthropic_api_key="")
    )
    monkeypatch.setattr(
        "agents.strategy_engine.get_settings", lambda: FakeSettings(anthropic_api_key="")
    )
    await seed_bars(session_factory, ticker="SPY", n=60)

    resp = await client.post("/api/pipeline/run/SPY")
    assert resp.status_code == 200
    body = resp.json()

    assert body["ticker"] == "SPY"
    assert body["signal"]["action"] in {"BUY", "SELL", "HOLD"}
    assert "reasoning" in body["signal"]
    assert "technical_profile" in body
    assert "sentiment_profile" in body
    assert "research" in body

    # Signal should have been persisted
    sig_resp = await client.get("/api/signals/SPY")
    assert sig_resp.json()["count"] == 1


@pytest.mark.asyncio
async def test_run_pipeline_without_ohlcv_data_returns_error(client):
    resp = await client.post("/api/pipeline/run/NODATA")
    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body


@pytest.mark.asyncio
async def test_run_pipeline_ignores_minute_bars(client, session_factory):
    """Indicators assume one bar per day; minute bars for a ticker with no
    daily history must not be picked up as a substitute."""
    async with session_factory() as session:
        session.add(OHLCV(
            ticker="MINUTEONLY",
            interval="minute",
            timestamp=datetime.now(),
            open=100, high=101, low=99, close=100.5, volume=1000,
        ))
        await session.commit()

    resp = await client.post("/api/pipeline/run/MINUTEONLY")
    assert resp.status_code == 200
    assert "error" in resp.json()


def _fake_finnhub_article(url="https://example.com/real-article", external_id=101):
    return {
        "title": "Real headline",
        "source": "Reuters",
        "url": url,
        "summary": "Something happened",
        "category": "company",
        "published_at": datetime.now(),
        "relevance": 0.9,
        "external_id": external_id,
    }


@pytest.mark.asyncio
async def test_run_pipeline_mock_mode_does_not_persist_articles(client, session_factory, monkeypatch):
    # autouse fixture already forces mock mode (finnhub_api_key="")
    monkeypatch.setattr("agents.sentiment_analyst.get_settings", lambda: FakeSettings(anthropic_api_key=""))
    monkeypatch.setattr("agents.strategy_engine.get_settings", lambda: FakeSettings(anthropic_api_key=""))
    await seed_bars(session_factory, ticker="SPY", n=60)

    resp = await client.post("/api/pipeline/run/SPY")
    assert resp.status_code == 200

    async with session_factory() as session:
        result = await session.execute(select(NewsArticle).where(NewsArticle.ticker == "SPY"))
        assert result.scalars().all() == []


@pytest.mark.asyncio
async def test_run_pipeline_live_mode_persists_articles_and_sentiment(client, session_factory, monkeypatch):
    monkeypatch.setattr("routers.pipeline.get_settings", lambda: FakeSettings(finnhub_api_key="fh-test"))
    monkeypatch.setattr("agents.sentiment_analyst.get_settings", lambda: FakeSettings(anthropic_api_key=""))
    monkeypatch.setattr("agents.strategy_engine.get_settings", lambda: FakeSettings(anthropic_api_key=""))

    async def fake_fetch(ticker, **kwargs):
        return [_fake_finnhub_article()]

    monkeypatch.setattr("data_sources.finnhub.fetch_company_news", fake_fetch)
    await seed_bars(session_factory, ticker="SPY", n=60)

    resp = await client.post("/api/pipeline/run/SPY")
    assert resp.status_code == 200
    body = resp.json()
    assert body["research"]["articles"][0]["title"] == "Real headline"

    async with session_factory() as session:
        result = await session.execute(select(NewsArticle).where(NewsArticle.ticker == "SPY"))
        rows = result.scalars().all()

    assert len(rows) == 1
    assert rows[0].url == "https://example.com/real-article"
    assert rows[0].external_id == 101
    assert rows[0].sentiment is not None
    assert rows[0].sentiment_scored_at is not None


@pytest.mark.asyncio
async def test_run_pipeline_second_live_run_skips_claude_for_cached_articles(client, session_factory, monkeypatch):
    monkeypatch.setattr("routers.pipeline.get_settings", lambda: FakeSettings(finnhub_api_key="fh-test"))
    monkeypatch.setattr(
        "agents.sentiment_analyst.get_settings", lambda: FakeSettings(anthropic_api_key="sk-ant-valid")
    )
    # Keep strategy_engine's reasoning off Claude so the call-count assertion
    # below reflects only sentiment_analyst's calls.
    monkeypatch.setattr("agents.strategy_engine.get_settings", lambda: FakeSettings(anthropic_api_key=""))

    fake_response = MagicMock()
    fake_response.content = [MagicMock(
        type="text",
        text='[{"index": 1, "sentiment": 0.5, "source_credibility": 0.9, '
             '"expected_impact": "medium", "reasoning": "ok"}]',
    )]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    monkeypatch.setattr("anthropic.Anthropic", lambda api_key: fake_client)

    async def fake_fetch(ticker, **kwargs):
        return [_fake_finnhub_article()]

    monkeypatch.setattr("data_sources.finnhub.fetch_company_news", fake_fetch)
    await seed_bars(session_factory, ticker="SPY", n=60)

    resp1 = await client.post("/api/pipeline/run/SPY")
    assert resp1.status_code == 200
    assert fake_client.messages.create.call_count == 1

    resp2 = await client.post("/api/pipeline/run/SPY")
    assert resp2.status_code == 200
    assert fake_client.messages.create.call_count == 1  # unchanged — article was cached, not re-scored
