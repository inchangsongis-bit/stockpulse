import pytest


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
