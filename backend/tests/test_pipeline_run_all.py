import pytest
from sqlalchemy import select

from models import Signal
from services import bulk_pipeline
from tests.test_bulk_pipeline import seed_bars, seed_watchlist
from tests.test_pipeline_api import FakeSettings


@pytest.fixture(autouse=True)
def _redirect_to_test_db(monkeypatch, session_factory):
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
async def test_run_all_endpoint_runs_pipeline_for_watchlist(client, session_factory):
    await seed_watchlist(session_factory, ["AAA", "BBB"])
    await seed_bars(session_factory, "AAA")
    await seed_bars(session_factory, "BBB")

    resp = await client.post("/api/pipeline/run-all")
    assert resp.status_code == 200
    assert resp.json() == {"status": "started"}

    # httpx's ASGI test transport runs BackgroundTasks in-process before
    # the request completes, so the run has already finished here.
    status_resp = await client.get("/api/pipeline/run-all/status")
    body = status_resp.json()
    assert body["running"] is False
    assert body["total"] == 2
    assert body["completed"] == 2

    async with session_factory() as session:
        result = await session.execute(select(Signal.ticker))
        signaled = {row[0] for row in result.all()}
    assert signaled == {"AAA", "BBB"}


@pytest.mark.asyncio
async def test_run_all_endpoint_reports_already_running(client):
    bulk_pipeline._status["running"] = True

    resp = await client.post("/api/pipeline/run-all")
    assert resp.status_code == 200
    assert resp.json()["status"] == "already_running"
