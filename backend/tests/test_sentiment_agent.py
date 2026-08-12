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
        "url": "https://example.com/beats-earnings",
        "summary": "Strong quarter with record growth",
        "relevance": 0.9,
    },
    {
        "title": "Analysts warn of slowdown risk",
        "source": "CNBC",
        "url": "https://example.com/slowdown-risk",
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
    for s in profile["article_scores"]:
        assert s["url"]  # every score carries the article's url for downstream matching
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
    fake_response.content = [MagicMock(type="text", text="not valid json")]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    monkeypatch.setattr("anthropic.Anthropic", lambda api_key: fake_client)

    agent = SentimentAnalystAgent()
    profile = await agent.run("SPY", {"research": {"articles": ARTICLES, "date": "2024-01-01"}})

    assert len(profile["article_scores"]) == 2


@pytest.mark.asyncio
async def test_claude_sentiment_skips_thinking_block_and_strips_markdown_fence(monkeypatch):
    """Regression test: models with extended thinking put a non-text
    'thinking' block at content[0], and often wrap JSON replies in a
    ```json fence even when told not to. Both must be handled."""
    monkeypatch.setattr(
        "agents.sentiment_analyst.get_settings",
        lambda: FakeSettings(anthropic_api_key="sk-ant-valid"),
    )

    json_reply = (
        '```json\n'
        '[{"index": 1, "sentiment": 0.6, "source_credibility": 0.9, '
        '"expected_impact": "high", "reasoning": "Beat estimates"},'
        '{"index": 2, "sentiment": -0.4, "source_credibility": 0.8, '
        '"expected_impact": "medium", "reasoning": "Slowdown risk"}]\n'
        '```'
    )
    thinking_block = MagicMock(type="thinking", text=None)
    text_block = MagicMock(type="text", text=json_reply)
    fake_response = MagicMock()
    fake_response.content = [thinking_block, text_block]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    monkeypatch.setattr("anthropic.Anthropic", lambda api_key: fake_client)

    agent = SentimentAnalystAgent()
    profile = await agent.run("SPY", {"research": {"articles": ARTICLES, "date": "2024-01-01"}})

    scores = {s["title"]: s["sentiment"] for s in profile["article_scores"]}
    assert scores["Company beats earnings expectations"] == 0.6
    assert scores["Analysts warn of slowdown risk"] == -0.4


@pytest.mark.asyncio
async def test_cached_sentiment_skips_claude_entirely_when_all_articles_cached(monkeypatch):
    monkeypatch.setattr(
        "agents.sentiment_analyst.get_settings",
        lambda: FakeSettings(anthropic_api_key="sk-ant-valid"),
    )
    fake_client = MagicMock()
    monkeypatch.setattr("anthropic.Anthropic", lambda api_key: fake_client)

    cached_sentiment = {
        a["url"]: {
            "title": a["title"], "source": a["source"], "url": a["url"],
            "sentiment": 0.5, "source_credibility": 0.9,
            "expected_impact": "medium", "reasoning": "cached reasoning",
        }
        for a in ARTICLES
    }

    agent = SentimentAnalystAgent()
    profile = await agent.run("SPY", {
        "research": {"articles": ARTICLES, "date": "2024-01-01"},
        "cached_sentiment": cached_sentiment,
    })

    fake_client.messages.create.assert_not_called()
    assert len(profile["article_scores"]) == 2
    for s in profile["article_scores"]:
        assert s["sentiment"] == 0.5
        assert s["reasoning"] == "cached reasoning"
    # relevance is refreshed from the current article data, not cached
    scores_by_title = {s["title"]: s["relevance"] for s in profile["article_scores"]}
    assert scores_by_title["Company beats earnings expectations"] == 0.9
    assert scores_by_title["Analysts warn of slowdown risk"] == 0.7


@pytest.mark.asyncio
async def test_cached_sentiment_only_scores_uncached_articles(monkeypatch):
    monkeypatch.setattr(
        "agents.sentiment_analyst.get_settings",
        lambda: FakeSettings(anthropic_api_key="sk-ant-valid"),
    )

    # Only the second article needs scoring — Claude's response is keyed
    # 1-based against whatever subset it was actually sent.
    json_reply = (
        '[{"index": 1, "sentiment": -0.4, "source_credibility": 0.8, '
        '"expected_impact": "medium", "reasoning": "Slowdown risk"}]'
    )
    fake_response = MagicMock()
    fake_response.content = [MagicMock(type="text", text=json_reply)]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    monkeypatch.setattr("anthropic.Anthropic", lambda api_key: fake_client)

    cached_sentiment = {
        ARTICLES[0]["url"]: {
            "title": ARTICLES[0]["title"], "source": ARTICLES[0]["source"], "url": ARTICLES[0]["url"],
            "sentiment": 0.5, "source_credibility": 0.9,
            "expected_impact": "medium", "reasoning": "cached reasoning",
        }
    }

    agent = SentimentAnalystAgent()
    profile = await agent.run("SPY", {
        "research": {"articles": ARTICLES, "date": "2024-01-01"},
        "cached_sentiment": cached_sentiment,
    })

    fake_client.messages.create.assert_called_once()
    prompt_sent = fake_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Analysts warn of slowdown risk" in prompt_sent
    assert "Company beats earnings expectations" not in prompt_sent  # cached article wasn't re-sent to Claude

    assert len(profile["article_scores"]) == 2
    scores_by_title = {s["title"]: s["sentiment"] for s in profile["article_scores"]}
    assert scores_by_title["Company beats earnings expectations"] == 0.5   # from cache
    assert scores_by_title["Analysts warn of slowdown risk"] == -0.4        # freshly scored
