import pytest

from models import Subscriber, Watchlist
from services import bulk_pipeline, daily_digest
from tests.test_bulk_pipeline import seed_bars
from tests.test_pipeline_api import FakeSettings


async def seed_watchlist(session_factory, tickers):
    async with session_factory() as session:
        for t in tickers:
            session.add(Watchlist(ticker=t))
        await session.commit()


async def seed_subscriber(session_factory, email, active=True):
    async with session_factory() as session:
        session.add(Subscriber(email=email, unsubscribe_token=f"tok-{email}", is_active=active))
        await session.commit()


@pytest.fixture(autouse=True)
def _redirect_to_test_db(monkeypatch, session_factory):
    monkeypatch.setattr("services.daily_digest.async_session", session_factory)
    monkeypatch.setattr("services.bulk_pipeline.async_session", session_factory)


@pytest.fixture(autouse=True)
def _mock_news(monkeypatch):
    monkeypatch.setattr("services.pipeline_runner.get_settings", lambda: FakeSettings(finnhub_api_key=""))


@pytest.fixture(autouse=True)
def _reset_bulk_status():
    def _clear():
        bulk_pipeline._status.update(
            running=False, trigger=None, total=0, completed=0,
            current_ticker=None, started_at=None, finished_at=None, errors={},
        )
    _clear()
    yield
    _clear()


@pytest.mark.asyncio
async def test_send_daily_digest_syncs_runs_pipeline_and_emails_active_subscribers(
    session_factory, monkeypatch
):
    await seed_watchlist(session_factory, ["AAA", "BBB"])
    await seed_bars(session_factory, "AAA")
    await seed_bars(session_factory, "BBB")
    await seed_subscriber(session_factory, "active@example.com", active=True)
    await seed_subscriber(session_factory, "inactive@example.com", active=False)

    sync_calls = []

    async def fake_sync(ticker, interval, days, db):
        sync_calls.append(ticker)
        return {"ticker": ticker}

    monkeypatch.setattr("services.daily_digest.sync_ticker_ohlcv", fake_sync)

    sent_to = []

    async def fake_send_email(to, subject, html):
        sent_to.append(to)
        return {"id": "fake"}

    monkeypatch.setattr("services.daily_digest.send_email", fake_send_email)

    result = await daily_digest.send_daily_digest()

    assert set(sync_calls) == {"AAA", "BBB"}
    assert sent_to == ["active@example.com"]  # inactive subscriber skipped
    assert result["subscribers"] == 1
    assert result["sent"] == 1
    assert result["failed"] == 0


@pytest.mark.asyncio
async def test_send_daily_digest_skips_a_ticker_whose_sync_fails(session_factory, monkeypatch):
    await seed_watchlist(session_factory, ["AAA", "BBB"])
    await seed_bars(session_factory, "AAA")
    await seed_bars(session_factory, "BBB")

    from data_sources.polygon import PolygonError

    async def fake_sync(ticker, interval, days, db):
        if ticker == "AAA":
            raise PolygonError("boom")
        return {"ticker": ticker}

    monkeypatch.setattr("services.daily_digest.sync_ticker_ohlcv", fake_sync)

    async def fake_send_email(to, subject, html):
        return {"id": "fake"}

    monkeypatch.setattr("services.daily_digest.send_email", fake_send_email)

    # Should not raise despite AAA's sync failing.
    result = await daily_digest.send_daily_digest()
    assert result["tickers"] == 2


@pytest.mark.asyncio
async def test_send_daily_digest_counts_failed_sends_without_stopping(session_factory, monkeypatch):
    await seed_watchlist(session_factory, ["AAA"])
    await seed_bars(session_factory, "AAA")
    await seed_subscriber(session_factory, "good@example.com")
    await seed_subscriber(session_factory, "bad@example.com")

    async def fake_sync(ticker, interval, days, db):
        return {"ticker": ticker}

    monkeypatch.setattr("services.daily_digest.sync_ticker_ohlcv", fake_sync)

    from services.email_sender import EmailError

    async def fake_send_email(to, subject, html):
        if to == "bad@example.com":
            raise EmailError("bounced")
        return {"id": "fake"}

    monkeypatch.setattr("services.daily_digest.send_email", fake_send_email)

    result = await daily_digest.send_daily_digest()
    assert result["sent"] == 1
    assert result["failed"] == 1


@pytest.mark.asyncio
async def test_send_daily_digest_with_no_subscribers_sends_nothing(session_factory, monkeypatch):
    await seed_watchlist(session_factory, ["AAA"])
    await seed_bars(session_factory, "AAA")

    async def fake_sync(ticker, interval, days, db):
        return {"ticker": ticker}

    monkeypatch.setattr("services.daily_digest.sync_ticker_ohlcv", fake_sync)

    called = False

    async def fake_send_email(to, subject, html):
        nonlocal called
        called = True

    monkeypatch.setattr("services.daily_digest.send_email", fake_send_email)

    result = await daily_digest.send_daily_digest()
    assert called is False
    assert result["sent"] == 0
