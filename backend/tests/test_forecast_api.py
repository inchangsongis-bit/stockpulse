from datetime import datetime, timedelta

import pytest

from analysis.forecast import ForecastUnavailable
from models import OHLCV, Signal


async def seed_minute_bars(session_factory, ticker="SPY", n=30):
    async with session_factory() as session:
        base = datetime.now() - timedelta(minutes=n)
        price = 100.0
        for i in range(n):
            price += 0.05
            session.add(OHLCV(
                ticker=ticker,
                interval="minute",
                timestamp=base + timedelta(minutes=i),
                open=price, high=price * 1.001, low=price * 0.999, close=price,
                volume=100_000,
            ))
        await session.commit()


async def seed_signal_with_sentiment(session_factory, ticker, sentiment_score):
    import json
    async with session_factory() as session:
        session.add(Signal(
            ticker=ticker,
            timestamp=datetime.now(),
            action="HOLD",
            confidence=50,
            reasoning="test",
            entry_low=1, entry_high=1, target=1, stop_loss=1,
            time_horizon="2-4 weeks", risk_level="low",
            factors_json=json.dumps({"sentiment": {"score": sentiment_score, "weight": 0.3, "weighted": 0.0}}),
        ))
        await session.commit()


@pytest.mark.asyncio
async def test_forecast_returns_prediction_for_ticker_with_enough_minute_data(client, session_factory, monkeypatch):
    await seed_minute_bars(session_factory, ticker="SPY", n=30)

    captured = {}

    def fake_predict(bars, sentiment=0.0):
        captured["n_bars"] = len(bars)
        captured["sentiment"] = sentiment
        return {"direction": "up", "probability_up": 0.62, "model_probability_up": 0.6, "confidence": 24.0, "horizon_minutes": 5}

    monkeypatch.setattr("routers.forecast.predict_direction", fake_predict)

    resp = await client.get("/api/forecast/SPY")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "SPY"
    assert body["direction"] == "up"
    assert body["probability_up"] == 0.62
    assert "as_of" in body
    assert captured["n_bars"] == 30
    assert captured["sentiment"] == 0.0  # no Signal seeded — defaults to 0.0


@pytest.mark.asyncio
async def test_forecast_passes_current_sentiment_from_latest_signal(client, session_factory, monkeypatch):
    await seed_minute_bars(session_factory, ticker="SPY", n=30)
    await seed_signal_with_sentiment(session_factory, "SPY", 0.42)

    captured = {}

    def fake_predict(bars, sentiment=0.0):
        captured["sentiment"] = sentiment
        return {"direction": "up", "probability_up": 0.55, "model_probability_up": 0.5, "confidence": 10.0, "horizon_minutes": 5}

    monkeypatch.setattr("routers.forecast.predict_direction", fake_predict)

    resp = await client.get("/api/forecast/SPY")
    assert resp.status_code == 200
    assert captured["sentiment"] == 0.42


@pytest.mark.asyncio
async def test_forecast_422s_when_not_enough_minute_bars(client, session_factory):
    await seed_minute_bars(session_factory, ticker="SPY", n=5)  # below _MIN_BARS

    resp = await client.get("/api/forecast/SPY")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_forecast_422s_when_no_minute_data_at_all(client):
    resp = await client.get("/api/forecast/NOPE")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_forecast_503s_when_model_not_trained(client, session_factory, monkeypatch):
    await seed_minute_bars(session_factory, ticker="SPY", n=30)

    def fake_predict(bars, sentiment=0.0):
        raise ForecastUnavailable("No trained forecast model found")

    monkeypatch.setattr("routers.forecast.predict_direction", fake_predict)

    resp = await client.get("/api/forecast/SPY")
    assert resp.status_code == 503
