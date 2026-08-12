from datetime import datetime, timedelta

import pytest

from data_sources.polygon import PolygonError
from tests.test_stocks_api import seed_bars


def fake_bars(ticker="SPY", n=3, interval="daily"):
    step = timedelta(days=1) if interval == "daily" else timedelta(minutes=1)
    base = datetime.now() - step * n
    return [
        {
            "ticker": ticker,
            "interval": interval,
            "timestamp": base + step * i,
            "open": 100 + i,
            "high": 101 + i,
            "low": 99 + i,
            "close": 100.5 + i,
            "volume": 1_000_000,
            "vwap": 100.4 + i,
            "num_trades": 1000,
        }
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_sync_replaces_existing_bars_with_live_data(client, session_factory, monkeypatch):
    await seed_bars(session_factory, ticker="SPY", n=5)  # stale/mock data already in DB

    async def fake_fetch(ticker, interval="daily", days=730):
        return fake_bars(ticker, n=3, interval=interval)

    monkeypatch.setattr("routers.stocks.fetch_ohlcv", fake_fetch)

    resp = await client.post("/api/stocks/SPY/sync")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "SPY"
    assert body["interval"] == "daily"
    assert body["synced_bars"] == 3
    assert body["latest_close"] == pytest.approx(102.5)

    # Old rows should be gone, replaced entirely by the 3 fresh bars
    ohlcv_resp = await client.get("/api/stocks/SPY/ohlcv?days=30")
    assert ohlcv_resp.json()["count"] == 3


@pytest.mark.asyncio
async def test_sync_minute_data_does_not_touch_daily_bars(client, session_factory, monkeypatch):
    await seed_bars(session_factory, ticker="SPY", n=5)  # existing daily bars

    async def fake_fetch(ticker, interval="daily", days=730):
        return fake_bars(ticker, n=10, interval=interval)

    monkeypatch.setattr("routers.stocks.fetch_ohlcv", fake_fetch)

    resp = await client.post("/api/stocks/SPY/sync?interval=minute")
    assert resp.status_code == 200
    assert resp.json()["interval"] == "minute"

    daily_resp = await client.get("/api/stocks/SPY/ohlcv?days=30&interval=daily")
    assert daily_resp.json()["count"] == 5  # untouched

    minute_resp = await client.get("/api/stocks/SPY/ohlcv?days=30&interval=minute")
    assert minute_resp.json()["count"] == 10


@pytest.mark.asyncio
async def test_sync_minute_days_capped_at_30(client):
    resp = await client.post("/api/stocks/SPY/sync?interval=minute&days=60")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_sync_returns_502_on_provider_error(client, monkeypatch):
    async def fake_fetch(ticker, interval="daily", days=730):
        raise PolygonError("boom")

    monkeypatch.setattr("routers.stocks.fetch_ohlcv", fake_fetch)

    resp = await client.post("/api/stocks/SPY/sync")
    assert resp.status_code == 502


@pytest.mark.asyncio
async def test_sync_returns_502_when_no_data_returned(client, monkeypatch):
    async def fake_fetch(ticker, interval="daily", days=730):
        return []

    monkeypatch.setattr("routers.stocks.fetch_ohlcv", fake_fetch)

    resp = await client.post("/api/stocks/SPY/sync")
    assert resp.status_code == 502
