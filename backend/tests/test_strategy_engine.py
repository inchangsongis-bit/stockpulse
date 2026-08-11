from dataclasses import dataclass
from unittest.mock import MagicMock

import httpx
import pytest
import anthropic

from agents.strategy_engine import StrategyEngineAgent


@dataclass
class FakeSettings:
    anthropic_api_key: str = ""


def technical(trend=0.0, momentum=0.0, volatility="unknown", volume_anomaly=False):
    return {
        "trend_score": trend,
        "momentum_score": momentum,
        "volatility_state": volatility,
        "volume_anomaly": volume_anomaly,
        "volume_anomaly_magnitude": 1.0,
        "support": 90.0,
        "resistance": 110.0,
        "patterns": [],
        "indicators": {"current_price": 100.0, "atr_14": 2.0, "rsi_14": 50.0, "macd_histogram": 0.0},
    }


def sentiment(score=0.0, trend="stable"):
    return {"composite_sentiment": score, "sentiment_trend": trend}


@pytest.mark.asyncio
async def test_strong_bullish_inputs_produce_buy(monkeypatch):
    monkeypatch.setattr(
        "agents.strategy_engine.get_settings", lambda: FakeSettings(anthropic_api_key="")
    )
    agent = StrategyEngineAgent()

    signal = await agent.run("SPY", {
        "technical_profile": technical(trend=1.0, momentum=1.0),
        "sentiment_profile": sentiment(score=0.8),
    })

    assert signal["action"] == "BUY"
    assert 0 <= signal["confidence"] <= 88


@pytest.mark.asyncio
async def test_strong_bearish_inputs_produce_sell(monkeypatch):
    monkeypatch.setattr(
        "agents.strategy_engine.get_settings", lambda: FakeSettings(anthropic_api_key="")
    )
    agent = StrategyEngineAgent()

    signal = await agent.run("SPY", {
        "technical_profile": technical(trend=-1.0, momentum=-1.0),
        "sentiment_profile": sentiment(score=-0.8),
    })

    assert signal["action"] == "SELL"


@pytest.mark.asyncio
async def test_conflicting_signals_force_hold(monkeypatch):
    monkeypatch.setattr(
        "agents.strategy_engine.get_settings", lambda: FakeSettings(anthropic_api_key="")
    )
    agent = StrategyEngineAgent()

    signal = await agent.run("SPY", {
        "technical_profile": technical(trend=0.9, momentum=0.5),   # bullish technicals
        "sentiment_profile": sentiment(score=-0.9),                # bearish sentiment
    })

    assert signal["action"] == "HOLD"
    assert "Conflicting signals" in signal["reasoning"]


@pytest.mark.asyncio
async def test_reasoning_falls_back_to_template_on_api_error(monkeypatch):
    monkeypatch.setattr(
        "agents.strategy_engine.get_settings",
        lambda: FakeSettings(anthropic_api_key="sk-ant-invalid"),
    )

    fake_request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    def raise_error(*args, **kwargs):
        raise anthropic.APIConnectionError(message="boom", request=fake_request)

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = raise_error
    monkeypatch.setattr("anthropic.Anthropic", lambda api_key: fake_client)

    agent = StrategyEngineAgent()
    signal = await agent.run("SPY", {
        "technical_profile": technical(trend=1.0, momentum=1.0),
        "sentiment_profile": sentiment(score=0.8),
    })

    # Falls back to template reasoning instead of raising
    assert signal["action"] == "BUY"
    assert "Signal: BUY" in signal["reasoning"]
    fake_client.messages.create.assert_called_once()


@pytest.mark.asyncio
async def test_reasoning_uses_claude_when_available(monkeypatch):
    monkeypatch.setattr(
        "agents.strategy_engine.get_settings",
        lambda: FakeSettings(anthropic_api_key="sk-ant-valid"),
    )

    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="Claude-generated reasoning paragraph.")]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    monkeypatch.setattr("anthropic.Anthropic", lambda api_key: fake_client)

    agent = StrategyEngineAgent()
    signal = await agent.run("SPY", {
        "technical_profile": technical(trend=1.0, momentum=1.0),
        "sentiment_profile": sentiment(score=0.8),
    })

    assert signal["reasoning"] == "Claude-generated reasoning paragraph."
