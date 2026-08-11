from dataclasses import dataclass
from unittest.mock import MagicMock

import httpx
import pytest
import anthropic

from agents.sentiment_analyst import SentimentAnalystAgent


@dataclass
class FakeSettings:
    anthropic_api_key: str = ""


ARTICLES = [
    {
        "title": "Company beats earnings expectations",
        "source": "Reuters",
        "summary": "Strong quarter with record growth",
        "relevance": 0.9,
    },
    {
        "title": "Analysts warn of slowdown risk",
        "source": "CNBC",
        "summary": "Concerns about weak demand and recession",
        "relevance": 0.7,
    },
]


@pytest.mark.asyncio
async def test_run_with_no_api_key_uses_rule_based_sentiment(monkeypatch):
    monkeypatch.setattr(
        "agents.sentiment_analyst.get_settings", lambda: FakeSettings(anthropic_api_key="")
    )
    agent = SentimentAnalystAgent()

    profile = await agent.run("SPY", {"research": {"articles": ARTICLES, "date": "2024-01-01"}})

    assert profile["ticker"] == "SPY"
    assert len(profile["article_scores"]) == 2
    # bullish article should score above the bearish one
    scores_by_title = {s["title"]: s["sentiment"] for s in profile["article_scores"]}
    assert scores_by_title["Company beats earnings expectations"] > scores_by_title["Analysts warn of slowdown risk"]


@pytest.mark.asyncio
async def test_run_with_no_articles_returns_empty_profile(monkeypatch):
    monkeypatch.setattr(
        "agents.sentiment_analyst.get_settings", lambda: FakeSettings(anthropic_api_key="")
    )
    agent = SentimentAnalystAgent()

    profile = await agent.run("SPY", {"research": {"articles": []}})

    assert profile["composite_sentiment"] == 0.0
    assert profile["article_scores"] == []


@pytest.mark.asyncio
async def test_claude_sentiment_falls_back_to_rules_on_api_error(monkeypatch):
    monkeypatch.setattr(
        "agents.sentiment_analyst.get_settings",
        lambda: FakeSettings(anthropic_api_key="sk-ant-invalid"),
    )

    fake_request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    def raise_auth_error(*args, **kwargs):
        raise anthropic.APIConnectionError(message="boom", request=fake_request)

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = raise_auth_error
    monkeypatch.setattr("anthropic.Anthropic", lambda api_key: fake_client)

    agent = SentimentAnalystAgent()
    profile = await agent.run("SPY", {"research": {"articles": ARTICLES, "date": "2024-01-01"}})

    # Should not raise — falls back to rule-based scoring instead
    assert len(profile["article_scores"]) == 2
    fake_client.messages.create.assert_called_once()


@pytest.mark.asyncio
async def test_claude_sentiment_falls_back_to_rules_on_malformed_response(monkeypatch):
    monkeypatch.setattr(
        "agents.sentiment_analyst.get_settings",
        lambda: FakeSettings(anthropic_api_key="sk-ant-valid"),
    )

    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="not valid json")]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    monkeypatch.setattr("anthropic.Anthropic", lambda api_key: fake_client)

    agent = SentimentAnalystAgent()
    profile = await agent.run("SPY", {"research": {"articles": ARTICLES, "date": "2024-01-01"}})

    assert len(profile["article_scores"]) == 2
