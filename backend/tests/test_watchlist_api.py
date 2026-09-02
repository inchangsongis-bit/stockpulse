from datetime import datetime, timedelta

import pytest

from models import OHLCV, Signal


async def seed_bar(session_factory, ticker, close, days_ago=0):
    async with session_factory() as session:
        session.add(OHLCV(
            ticker=ticker,
            timestamp=datetime.now() - timedelta(days=days_ago),
            open=close, high=close, low=close, close=close,
            volume=1_000_000,
        ))
        await session.commit()


async def seed_signal(session_factory, ticker, action, confidence, days_ago=0):
    async with session_factory() as session:
        session.add(Signal(
            ticker=ticker,
            timestamp=datetime.now() - timedelta(days=days_ago),
            action=action,
            confidence=confidence,
            reasoning="test",
            entry_low=1, entry_high=1, target=1, stop_loss=1,
            time_horizon="2-4 weeks", risk_level="low",
            factors_json="{}",
        ))
        await session.commit()


@pytest.mark.asyncio
async def test_summary_empty_watchlist(client):
    resp = await client.get("/api/watchlist/summary")
    assert resp.status_code == 200
    assert resp.json() == {"tickers": []}


@pytest.mark.asyncio
async def test_summary_includes_latest_price_and_signal(client, session_factory):
    await client.post("/api/watchlist/", json={"ticker": "AAPL"})

    await seed_bar(session_factory, "AAPL", close=100.0, days_ago=2)
    await seed_bar(session_factory, "AAPL", close=105.0, days_ago=0)  # latest
    await seed_signal(session_factory, "AAPL", "SELL", 40, days_ago=3)
    await seed_signal(session_factory, "AAPL", "BUY", 60, days_ago=0)  # latest

    resp = await client.get("/api/watchlist/summary")
    assert resp.status_code == 200
    body = resp.json()["tickers"]
    assert len(body) == 1
    assert body[0]["ticker"] == "AAPL"
    assert body[0]["price"] == 105.0
    assert body[0]["signal"]["action"] == "BUY"
    assert body[0]["signal"]["confidence"] == 60


@pytest.mark.asyncio
async def test_summary_signal_is_null_when_ticker_never_pipelined(client, session_factory):
    await client.post("/api/watchlist/", json={"ticker": "HSBC"})
    await seed_bar(session_factory, "HSBC", close=50.0)

    resp = await client.get("/api/watchlist/summary")
    body = resp.json()["tickers"]
    assert body[0]["signal"] is None
    assert body[0]["price"] == 50.0


@pytest.mark.asyncio
async def test_add_ticker(client):
    resp = await client.post("/api/watchlist/", json={"ticker": "aapl"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "added", "ticker": "AAPL"}

    resp = await client.get("/api/watchlist/")
    tickers = [t["ticker"] for t in resp.json()["tickers"]]
    assert tickers == ["AAPL"]


@pytest.mark.asyncio
async def test_add_duplicate_ticker_conflicts(client):
    await client.post("/api/watchlist/", json={"ticker": "MSFT"})
    resp = await client.post("/api/watchlist/", json={"ticker": "MSFT"})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_add_empty_ticker_rejected(client):
    resp = await client.post("/api/watchlist/", json={"ticker": "   "})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_remove_ticker(client):
    await client.post("/api/watchlist/", json={"ticker": "TSLA"})
    resp = await client.delete("/api/watchlist/TSLA")
    assert resp.status_code == 200
    assert resp.json() == {"status": "removed", "ticker": "TSLA"}

    resp = await client.get("/api/watchlist/")
    assert resp.json()["tickers"] == []


@pytest.mark.asyncio
async def test_remove_nonexistent_ticker_404s(client):
    resp = await client.delete("/api/watchlist/NOPE")
    assert resp.status_code == 404
