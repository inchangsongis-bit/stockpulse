from datetime import datetime, timedelta

import pytest

from models import OHLCV


async def seed_bars(session_factory, ticker="SPY", n=5, start_days_ago=4):
    async with session_factory() as session:
        base = datetime.now() - timedelta(days=start_days_ago)
        for i in range(n):
            session.add(OHLCV(
                ticker=ticker,
                timestamp=base + timedelta(days=i),
                open=100 + i,
                high=101 + i,
                low=99 + i,
                close=100.5 + i,
                volume=1_000_000 + i,
                vwap=100.4 + i,
            ))
        await session.commit()


@pytest.mark.asyncio
async def test_ohlcv_returns_seeded_bars(client, session_factory):
    await seed_bars(session_factory, ticker="SPY", n=5)

    resp = await client.get("/api/stocks/SPY/ohlcv?days=30")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "SPY"
    assert body["count"] == 5
    assert len(body["data"]) == 5
    # ascending order
    timestamps = [row["timestamp"] for row in body["data"]]
    assert timestamps == sorted(timestamps)


@pytest.mark.asyncio
async def test_ohlcv_is_case_insensitive_and_scoped_to_ticker(client, session_factory):
    await seed_bars(session_factory, ticker="SPY", n=3)
    await seed_bars(session_factory, ticker="AAPL", n=2)

    resp = await client.get("/api/stocks/spy/ohlcv?days=30")
    assert resp.status_code == 200
    assert resp.json()["count"] == 3


@pytest.mark.asyncio
async def test_ohlcv_respects_days_window(client, session_factory):
    await seed_bars(session_factory, ticker="SPY", n=5, start_days_ago=4)

    resp = await client.get("/api/stocks/SPY/ohlcv?days=1")
    assert resp.status_code == 200
    # only bars from the last day should be included
    assert resp.json()["count"] < 5


@pytest.mark.asyncio
async def test_ohlcv_unknown_ticker_returns_empty(client):
    resp = await client.get("/api/stocks/NOPE/ohlcv?days=90")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["data"] == []


@pytest.mark.asyncio
async def test_latest_returns_most_recent_bar(client, session_factory):
    await seed_bars(session_factory, ticker="SPY", n=5)

    resp = await client.get("/api/stocks/SPY/latest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "SPY"
    assert body["close"] == pytest.approx(104.5)  # last of 5 bars: 100.5 + 4


@pytest.mark.asyncio
async def test_latest_with_no_data_returns_error(client):
    resp = await client.get("/api/stocks/NOPE/latest")
    assert resp.status_code == 200
    assert resp.json() == {"error": "No data found"}
