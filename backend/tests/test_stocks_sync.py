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
async def test_sync_first_time_fetches_full_window(client, monkeypatch):
    async def fake_fetch(ticker, interval="daily", days=730):
        assert days == 730  # no prior data — bootstrap uses the full default window
        return fake_bars(ticker, n=3, interval=interval)

    monkeypatch.setattr("services.ohlcv_sync.fetch_ohlcv", fake_fetch)

    resp = await client.post("/api/stocks/SPY/sync")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "SPY"
    assert body["interval"] == "daily"
    assert body["mode"] == "full"
    assert body["synced_bars"] == 3
    assert body["latest_close"] == pytest.approx(102.5)

    ohlcv_resp = await client.get("/api/stocks/SPY/ohlcv?days=30")
    assert ohlcv_resp.json()["count"] == 3


@pytest.mark.asyncio
async def test_sync_incremental_preserves_older_history(client, session_factory, monkeypatch):
    # Bars already stored from 30 days ago through today — a ticker
    # that's been synced before.
    await seed_bars(session_factory, ticker="SPY", n=31, start_days_ago=30)

    fresh = fake_bars("SPY", n=3, interval="daily")  # spans ~now-3..now-1
    captured = {}

    async def fake_fetch(ticker, interval="daily", days=730):
        captured["days"] = days
        return fresh

    monkeypatch.setattr("services.ohlcv_sync.fetch_ohlcv", fake_fetch)

    # A short display range pill must not shrink the fetch window or
    # wipe older history.
    resp = await client.post("/api/stocks/SPY/sync?days=30")
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "incremental"
    assert body["synced_bars"] == 3

    # The incremental fetch window should be small (near the overlap
    # buffer for daily bars), not the full 730-day default.
    assert captured["days"] <= 10

    ohlcv_resp = await client.get("/api/stocks/SPY/ohlcv?days=60")
    data = ohlcv_resp.json()["data"]
    # The oldest seeded bar (well before the fetch/overlap window) must
    # have survived the sync untouched.
    assert any(d["close"] == pytest.approx(100.5) for d in data)


@pytest.mark.asyncio
async def test_sync_incremental_ignores_days_param_for_bootstrap_window(client, session_factory, monkeypatch):
    await seed_bars(session_factory, ticker="SPY", n=2, start_days_ago=1)

    async def fake_fetch(ticker, interval="daily", days=730):
        # days must be 400 for a first-ever sync to actually be a bug —
        # here it should stay small since a bar already exists.
        assert days <= 10
        return fake_bars("SPY", n=1, interval=interval)

    monkeypatch.setattr("services.ohlcv_sync.fetch_ohlcv", fake_fetch)

    resp = await client.post("/api/stocks/SPY/sync?days=400")
    assert resp.status_code == 200
    assert resp.json()["mode"] == "incremental"


@pytest.mark.asyncio
async def test_sync_minute_data_does_not_touch_daily_bars(client, session_factory, monkeypatch):
    await seed_bars(session_factory, ticker="SPY", n=5)  # existing daily bars

    async def fake_fetch(ticker, interval="daily", days=730):
        return fake_bars(ticker, n=10, interval=interval)

    monkeypatch.setattr("services.ohlcv_sync.fetch_ohlcv", fake_fetch)

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

    monkeypatch.setattr("services.ohlcv_sync.fetch_ohlcv", fake_fetch)

    resp = await client.post("/api/stocks/SPY/sync")
    assert resp.status_code == 502


@pytest.mark.asyncio
async def test_sync_returns_502_when_no_data_returned(client, monkeypatch):
    async def fake_fetch(ticker, interval="daily", days=730):
        return []

    monkeypatch.setattr("services.ohlcv_sync.fetch_ohlcv", fake_fetch)

    resp = await client.post("/api/stocks/SPY/sync")
    assert resp.status_code == 502
