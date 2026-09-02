import pytest
from sqlalchemy import select

from models import Subscriber


@pytest.mark.asyncio
async def test_subscribe_adds_a_new_subscriber(client):
    resp = await client.post("/api/subscribers/", json={"email": "Someone@Example.com"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "subscribed", "email": "someone@example.com"}

    resp = await client.get("/api/subscribers/")
    emails = [s["email"] for s in resp.json()["subscribers"]]
    assert emails == ["someone@example.com"]


@pytest.mark.asyncio
async def test_subscribe_rejects_invalid_email(client):
    resp = await client.post("/api/subscribers/", json={"email": "not-an-email"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_subscribe_twice_conflicts(client):
    await client.post("/api/subscribers/", json={"email": "a@example.com"})
    resp = await client.post("/api/subscribers/", json={"email": "a@example.com"})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_unsubscribe_via_token_deactivates_without_deleting(client, session_factory):
    await client.post("/api/subscribers/", json={"email": "a@example.com"})

    async with session_factory() as session:
        result = await session.execute(select(Subscriber).where(Subscriber.email == "a@example.com"))
        token = result.scalar_one().unsubscribe_token

    resp = await client.get(f"/api/subscribers/unsubscribe/{token}")
    assert resp.status_code == 200
    assert "Unsubscribed" in resp.text

    async with session_factory() as session:
        result = await session.execute(select(Subscriber).where(Subscriber.email == "a@example.com"))
        sub = result.scalar_one()
    assert sub.is_active is False
    assert sub.unsubscribed_at is not None


@pytest.mark.asyncio
async def test_unsubscribe_with_unknown_token_404s(client):
    resp = await client.get("/api/subscribers/unsubscribe/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_resubscribing_after_unsubscribe_reactivates_same_row(client, session_factory):
    await client.post("/api/subscribers/", json={"email": "a@example.com"})
    async with session_factory() as session:
        result = await session.execute(select(Subscriber).where(Subscriber.email == "a@example.com"))
        token = result.scalar_one().unsubscribe_token
    await client.get(f"/api/subscribers/unsubscribe/{token}")

    resp = await client.post("/api/subscribers/", json={"email": "a@example.com"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "resubscribed"

    async with session_factory() as session:
        result = await session.execute(select(Subscriber).where(Subscriber.email == "a@example.com"))
        rows = result.scalars().all()
    assert len(rows) == 1  # reactivated, not duplicated
    assert rows[0].is_active is True


@pytest.mark.asyncio
async def test_remove_subscriber(client):
    await client.post("/api/subscribers/", json={"email": "a@example.com"})
    resp = await client.delete("/api/subscribers/a@example.com")
    assert resp.status_code == 200

    resp = await client.get("/api/subscribers/")
    assert resp.json()["subscribers"] == []


@pytest.mark.asyncio
async def test_remove_nonexistent_subscriber_404s(client):
    resp = await client.delete("/api/subscribers/nope@example.com")
    assert resp.status_code == 404
