"""
Send a one-off test digest email using the CURRENT watchlist data,
without waiting for the scheduled job or running a fresh sync/pipeline
pass first. Requires RESEND_API_KEY to be set in the repo root's .env.

    cd backend && source venv/bin/activate && python scripts/send_test_digest.py someone@example.com

Note: without a verified sending domain in Resend, the default sandbox
sender (onboarding@resend.dev) can only deliver to the email address(es)
verified on your own Resend account — not to arbitrary recipients.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import async_session  # noqa: E402
from services.email_sender import EmailError, send_email  # noqa: E402
from services.email_templates import render_daily_digest_html  # noqa: E402
from services.watchlist_summary import get_summary_rows  # noqa: E402


async def main(to_email: str):
    async with async_session() as db:
        rows = await get_summary_rows(db)

    if not rows:
        print("Watchlist is empty — nothing to send.")
        return

    html = render_daily_digest_html(rows, as_of=datetime.now())
    print(f"Sending test digest ({len(rows)} tickers) to {to_email}...")
    try:
        result = await send_email(to_email, "StockPulse Daily Signals (test)", html)
    except EmailError as e:
        print(f"Failed to send: {e}")
        sys.exit(1)

    print(f"Sent. Resend response: {result}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/send_test_digest.py <email>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
