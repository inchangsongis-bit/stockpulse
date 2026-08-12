from datetime import datetime, timedelta
from dataclasses import dataclass

import pytest

from models import OHLCV


@dataclass
class FakeSettings:
    anthropic_api_key: str = ""


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
