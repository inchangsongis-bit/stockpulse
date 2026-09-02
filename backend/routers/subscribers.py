"""
Daily-digest email subscriber management. No auth system exists anywhere
in this prototype yet, so /  and DELETE are unauthenticated admin-ish
endpoints for now — fine for local/personal use, would need real auth
before this app had other people's data in it.
"""

import secrets
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Subscriber

router = APIRouter(prefix="/api/subscribers", tags=["subscribers"])


class SubscribeRequest(BaseModel):
    email: str


@router.get("/")
async def list_subscribers(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Subscriber).order_by(Subscriber.subscribed_at.desc()))
    rows = result.scalars().all()
    return {
        "subscribers": [
            {
                "email": r.email,
                "is_active": r.is_active,
                "subscribed_at": r.subscribed_at.isoformat() if r.subscribed_at else None,
                "unsubscribed_at": r.unsubscribed_at.isoformat() if r.unsubscribed_at else None,
            }
            for r in rows
        ]
    }


@router.post("/")
async def subscribe(req: SubscribeRequest, db: AsyncSession = Depends(get_db)):
    email = req.email.strip().lower()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Invalid email address")

    existing_result = await db.execute(select(Subscriber).where(Subscriber.email == email))
    existing = existing_result.scalar_one_or_none()

    if existing:
        if existing.is_active:
            raise HTTPException(status_code=409, detail=f"{email} is already subscribed")
        # Re-subscribing after a prior unsubscribe — reactivate rather
        # than create a duplicate row.
        existing.is_active = True
        existing.unsubscribed_at = None
        await db.commit()
        return {"status": "resubscribed", "email": email}

    entry = Subscriber(email=email, unsubscribe_token=secrets.token_urlsafe(32))
    db.add(entry)
    await db.commit()
    return {"status": "subscribed", "email": email}


@router.get("/unsubscribe/{token}")
async def unsubscribe(token: str, db: AsyncSession = Depends(get_db)):
    """
    Hit directly from the unsubscribe link in the digest email — no auth,
    just an unguessable per-subscriber token, so this has to stay a GET
    (email clients only ever generate link clicks, never POSTs).
    """
    result = await db.execute(select(Subscriber).where(Subscriber.unsubscribe_token == token))
    sub = result.scalar_one_or_none()
    if not sub:
        raise HTTPException(status_code=404, detail="Unknown unsubscribe link")

    sub.is_active = False
    sub.unsubscribed_at = datetime.now()
    await db.commit()

    return Response(
        content=(
            "<!doctype html><html><body style='font-family:sans-serif;padding:40px;text-align:center;'>"
            f"<h2>Unsubscribed</h2><p>{sub.email} will no longer receive the StockPulse daily digest.</p>"
            "</body></html>"
        ),
        media_type="text/html",
    )


@router.delete("/{email}")
async def remove_subscriber(email: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Subscriber).where(Subscriber.email == email.strip().lower()))
    sub = result.scalar_one_or_none()
    if not sub:
        raise HTTPException(status_code=404, detail=f"{email} not found")
    await db.delete(sub)
    await db.commit()
    return {"status": "removed", "email": email}
