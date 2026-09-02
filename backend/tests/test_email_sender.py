from dataclasses import dataclass

import pytest

from services.email_sender import EmailError, send_email


@dataclass
class FakeSettings:
    resend_api_key: str = "re-test-key"
    resend_from_address: str = "StockPulse <onboarding@resend.dev>"


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeAsyncClient:
    def __init__(self, response):
        self._response = response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self._response


@pytest.mark.asyncio
async def test_send_email_posts_to_resend_with_expected_payload(monkeypatch):
    monkeypatch.setattr("services.email_sender.get_settings", lambda: FakeSettings())
    fake_client = FakeAsyncClient(FakeResponse(200, {"id": "abc123"}))
    monkeypatch.setattr("services.email_sender.httpx.AsyncClient", lambda *a, **k: fake_client)

    result = await send_email("someone@example.com", "Subject line", "<p>Body</p>")

    assert result == {"id": "abc123"}
    url, kwargs = fake_client.calls[0]
    assert url == "https://api.resend.com/emails"
    assert kwargs["headers"]["Authorization"] == "Bearer re-test-key"
    assert kwargs["json"]["to"] == ["someone@example.com"]
    assert kwargs["json"]["subject"] == "Subject line"
    assert kwargs["json"]["html"] == "<p>Body</p>"
    assert kwargs["json"]["from"] == "StockPulse <onboarding@resend.dev>"


@pytest.mark.asyncio
async def test_send_email_raises_when_api_key_missing(monkeypatch):
    monkeypatch.setattr("services.email_sender.get_settings", lambda: FakeSettings(resend_api_key=""))

    with pytest.raises(EmailError):
        await send_email("someone@example.com", "Subject", "<p>Body</p>")


@pytest.mark.asyncio
async def test_send_email_raises_on_non_2xx_response(monkeypatch):
    monkeypatch.setattr("services.email_sender.get_settings", lambda: FakeSettings())
    fake_client = FakeAsyncClient(FakeResponse(422, text="invalid recipient"))
    monkeypatch.setattr("services.email_sender.httpx.AsyncClient", lambda *a, **k: fake_client)

    with pytest.raises(EmailError):
        await send_email("bad", "Subject", "<p>Body</p>")
