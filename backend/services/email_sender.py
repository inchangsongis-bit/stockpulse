"""
Sends email via Resend's API (https://resend.com/docs/api-reference/emails/send-email).
Mirrors the shape of data_sources/polygon.py and data_sources/finnhub.py: a
module-level Error class, one async send function that reads its key from
get_settings() itself, raises on a missing key or a non-2xx response.
"""

import httpx

from config import get_settings


class EmailError(Exception):
    pass


async def send_email(to: str, subject: str, html: str) -> dict:
    settings = get_settings()
    if not settings.resend_api_key:
        raise EmailError("RESEND_API_KEY is not configured")

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {settings.resend_api_key}"},
            json={
                "from": settings.resend_from_address,
                "to": [to],
                "subject": subject,
                "html": html,
            },
        )

    if resp.status_code >= 300:
        raise EmailError(f"Resend API error {resp.status_code}: {resp.text[:300]}")

    return resp.json()
