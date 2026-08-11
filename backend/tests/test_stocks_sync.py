from datetime import datetime, timedelta

import pytest

from data_sources.polygon import PolygonError
from tests.test_stocks_api import seed_bars


def fake_bars(ticker="SPY", n=3):
    base = datetime.now() - timedelta(days=n)
    return [
        {
            "ticker": ticker,
            "timestamp": base + timedelta(days=i),
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

    async def fake_fetch(ticker, days=730):
        return fake_bars(ticker, n=3)

    monkeypatch.setattr("routers.stocks.fetch_daily_ohlcv", fake_fetch)

    resp = await client.post("/api/stocks/SPY/sync")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "SPY"
    assert body["synced_bars"] == 3
    assert body["latest_close"] == pytest.approx(102.5)

    # Old rows should be gone, replaced entirely by the 3 fresh bars
    ohlcv_resp = await client.get("/api/stocks/SPY/ohlcv?days=30")
    assert ohlcv_resp.json()["count"] == 3


@pytest.mark.asyncio
async def test_sync_returns_502_on_provider_error(client, monkeypatch):
    async def fake_fetch(ticker, days=730):
        raise PolygonError("boom")

    monkeypatch.setattr("routers.stocks.fetch_daily_ohlcv", fake_fetch)

    resp = await client.post("/api/stocks/SPY/sync")
    assert resp.status_code == 502


@pytest.mark.asyncio
async def test_sync_returns_502_when_no_data_returned(client, monkeypatch):
    async def fake_fetch(ticker, days=730):
        return []

    monkeypatch.setattr("routers.stocks.fetch_daily_ohlcv", fake_fetch)

    resp = await client.post("/api/stocks/SPY/sync")
    assert resp.status_code == 502
