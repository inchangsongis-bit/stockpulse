from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from data_sources.finnhub import FinnhubError, fetch_company_news


@dataclass
class FakeSettings:
    finnhub_api_key: str = "fh-test-key"


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class FakeAsyncClient:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, *args, **kwargs):
        return self._response


def make_raw(headline="Some headline", hours_ago=1, source="Reuters", article_id=1, **overrides):
    dt = int((datetime.now() - timedelta(hours=hours_ago)).timestamp())
    row = {
        "category": "company",
        "datetime": dt,
        "headline": headline,
        "id": article_id,
        "image": "https://example.com/img.png",
        "related": "SPY",
        "source": source,
        "summary": "A summary.",
        "url": f"https://example.com/article-{article_id}",
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_fetch_company_news_maps_fields(monkeypatch):
    monkeypatch.setattr("data_sources.finnhub.get_settings", lambda: FakeSettings())
    raw = [make_raw(article_id=1)]
    monkeypatch.setattr(
        "data_sources.finnhub.httpx.AsyncClient",
        lambda *a, **k: FakeAsyncClient(FakeResponse(200, raw)),
    )

    articles = await fetch_company_news("SPY")

    assert len(articles) == 1
    a = articles[0]
    assert a["title"] == "Some headline"
    assert a["source"] == "Reuters"
    assert a["url"] == "https://example.com/article-1"
    assert a["external_id"] == 1
    assert 0.0 <= a["relevance"] <= 1.0


@pytest.mark.asyncio
async def test_fetch_company_news_filters_junk_rows(monkeypatch):
    monkeypatch.setattr("data_sources.finnhub.get_settings", lambda: FakeSettings())
    raw = [
        make_raw(article_id=1, headline=""),        # empty headline
        make_raw(article_id=2, datetime=0),           # zero timestamp
        make_raw(article_id=3),                       # valid
    ]
    monkeypatch.setattr(
        "data_sources.finnhub.httpx.AsyncClient",
        lambda *a, **k: FakeAsyncClient(FakeResponse(200, raw)),
    )

    articles = await fetch_company_news("SPY")

    assert len(articles) == 1
    assert articles[0]["external_id"] == 3


@pytest.mark.asyncio
async def test_fetch_company_news_orders_most_recent_first_and_respects_limit(monkeypatch):
    monkeypatch.setattr("data_sources.finnhub.get_settings", lambda: FakeSettings())
    raw = [make_raw(article_id=i, hours_ago=i) for i in range(1, 21)]  # 20 articles, 1..20 hours old
    monkeypatch.setattr(
        "data_sources.finnhub.httpx.AsyncClient",
        lambda *a, **k: FakeAsyncClient(FakeResponse(200, raw)),
    )

    articles = await fetch_company_news("SPY", limit=5)

    assert len(articles) == 5
    assert [a["external_id"] for a in articles] == [1, 2, 3, 4, 5]  # most recent (fewest hours_ago) first
    # most recent article should have the highest relevance
    assert articles[0]["relevance"] >= articles[-1]["relevance"]


@pytest.mark.asyncio
async def test_fetch_company_news_raises_without_api_key(monkeypatch):
    monkeypatch.setattr("data_sources.finnhub.get_settings", lambda: FakeSettings(finnhub_api_key=""))

    with pytest.raises(FinnhubError):
        await fetch_company_news("SPY")


@pytest.mark.asyncio
async def test_fetch_company_news_raises_on_non_200(monkeypatch):
    monkeypatch.setattr("data_sources.finnhub.get_settings", lambda: FakeSettings())
    monkeypatch.setattr(
        "data_sources.finnhub.httpx.AsyncClient",
        lambda *a, **k: FakeAsyncClient(FakeResponse(500, None, text="server error")),
    )

    with pytest.raises(FinnhubError):
        await fetch_company_news("SPY")


@pytest.mark.asyncio
async def test_fetch_company_news_raises_on_unexpected_payload(monkeypatch):
    monkeypatch.setattr("data_sources.finnhub.get_settings", lambda: FakeSettings())
    monkeypatch.setattr(
        "data_sources.finnhub.httpx.AsyncClient",
        lambda *a, **k: FakeAsyncClient(FakeResponse(200, {"error": "bad symbol"})),
    )

    with pytest.raises(FinnhubError):
        await fetch_company_news("SPY")
