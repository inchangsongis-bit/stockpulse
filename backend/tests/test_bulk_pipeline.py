from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from models import OHLCV, Signal, Watchlist
from services import bulk_pipeline
from tests.test_pipeline_api import FakeSettings


async def seed_watchlist(session_factory, tickers):
    async with session_factory() as session:
        for t in tickers:
            session.add(Watchlist(ticker=t))
        await session.commit()


async def seed_bars(session_factory, ticker, n=10):
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


@pytest.fixture(autouse=True)
def _redirect_to_test_db(monkeypatch, session_factory):
    # bulk_pipeline isn't request-scoped like the router endpoints — it
    # opens its own sessions via database.async_session, so redirect that
    # at the test's in-memory DB instead of the real app engine.
    monkeypatch.setattr("services.bulk_pipeline.async_session", session_factory)


@pytest.fixture(autouse=True)
def _mock_news(monkeypatch):
    monkeypatch.setattr("services.pipeline_runner.get_settings", lambda: FakeSettings(finnhub_api_key=""))


@pytest.fixture(autouse=True)
def _reset_status():
    def _clear():
        bulk_pipeline._status.update(
            running=False, trigger=None, total=0, completed=0,
            current_ticker=None, started_at=None, finished_at=None, errors={},
        )
    _clear()
    yield
    _clear()


@pytest.mark.asyncio
async def test_run_all_runs_pipeline_for_every_watchlist_ticker(session_factory):
    await seed_watchlist(session_factory, ["AAA", "BBB"])
    await seed_bars(session_factory, "AAA")
    await seed_bars(session_factory, "BBB")

    await bulk_pipeline.run_all(trigger="manual")

    status = bulk_pipeline.get_status()
    assert status["running"] is False
    assert status["trigger"] == "manual"
    assert status["total"] == 2
    assert status["completed"] == 2
    assert status["errors"] == {}
    assert status["finished_at"] is not None

    async with session_factory() as session:
        result = await session.execute(select(Signal.ticker))
        signaled = {row[0] for row in result.all()}
    assert signaled == {"AAA", "BBB"}


@pytest.mark.asyncio
async def test_run_all_records_per_ticker_errors_without_stopping(session_factory, monkeypatch):
    await seed_watchlist(session_factory, ["AAA", "BBB"])

    async def fake_runner(ticker, db):
        if ticker == "AAA":
            raise RuntimeError("boom")
        return {"ok": True}

    monkeypatch.setattr("services.bulk_pipeline.run_pipeline_for_ticker", fake_runner)

    await bulk_pipeline.run_all(trigger="manual")

    status = bulk_pipeline.get_status()
    assert status["completed"] == 2
    assert status["errors"] == {"AAA": "boom"}
    assert status["running"] is False


@pytest.mark.asyncio
async def test_run_all_noops_if_already_running():
    bulk_pipeline._status["running"] = True

    await bulk_pipeline.run_all(trigger="manual")

    # Never got past the guard, so nothing was set up for a real run.
    assert bulk_pipeline._status["total"] == 0
    assert bulk_pipeline._status["completed"] == 0
