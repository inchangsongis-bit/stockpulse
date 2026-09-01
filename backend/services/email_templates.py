"""
Builds the HTML for the daily BUY/SELL/HOLD digest email. Plain
table-based HTML with inline styles throughout (not <style> blocks) — most
email clients strip <style> tags or apply them inconsistently, so every
element carries its own styling directly, same constraint email templates
always work under.
"""

from datetime import datetime
from typing import List, Optional, TypedDict

API_BASE_UNSUBSCRIBE_URL = "http://localhost:8000/api/subscribers/unsubscribe"

_ACTION_COLORS = {
    "BUY": "#22c55e",
    "SELL": "#ef4444",
    "HOLD": "#eab308",
}


class DigestRow(TypedDict):
    ticker: str
    price: Optional[float]
    signal: Optional[dict]  # {"action": str, "confidence": int}


def _row_html(row: DigestRow) -> str:
    ticker = row["ticker"]
    price = f"${row['price']:.2f}" if row.get("price") is not None else "—"
    signal = row.get("signal")

    if signal:
        color = _ACTION_COLORS.get(signal["action"], "#888888")
        badge = (
            f'<span style="background:{color}22;color:{color};border:1px solid {color}55;'
            f'border-radius:4px;padding:2px 8px;font-weight:600;font-size:12px;">{signal["action"]}</span>'
        )
        confidence = f'<span style="color:#888888;font-size:12px;">{signal["confidence"]}%</span>'
    else:
        badge = '<span style="color:#888888;font-size:12px;">Not analyzed</span>'
        confidence = ""

    return f"""
    <tr>
      <td style="padding:8px 12px;border-bottom:1px solid #2a2a3a;font-weight:600;">{ticker}</td>
      <td style="padding:8px 12px;border-bottom:1px solid #2a2a3a;font-family:monospace;color:#cccccc;">{price}</td>
      <td style="padding:8px 12px;border-bottom:1px solid #2a2a3a;">{badge}</td>
      <td style="padding:8px 12px;border-bottom:1px solid #2a2a3a;">{confidence}</td>
    </tr>"""


def render_daily_digest_html(
    rows: List[DigestRow],
    as_of: datetime,
    unsubscribe_token: Optional[str] = None,
) -> str:
    """
    rows: watchlist summary rows (same shape as GET /api/watchlist/summary),
    already sorted/grouped however the caller wants them to appear.
    unsubscribe_token: this recipient's Subscriber.unsubscribe_token —
    omit for a test send that isn't tied to a real subscriber row.
    """
    buys = [r for r in rows if r.get("signal") and r["signal"]["action"] == "BUY"]
    sells = [r for r in rows if r.get("signal") and r["signal"]["action"] == "SELL"]
    holds = [r for r in rows if r.get("signal") and r["signal"]["action"] == "HOLD"]
    unrated = [r for r in rows if not r.get("signal")]

    def section(title: str, section_rows: List[DigestRow]) -> str:
        if not section_rows:
            return ""
        rows_html = "".join(_row_html(r) for r in section_rows)
        return f"""
        <tr><td style="padding:16px 12px 4px;font-size:14px;font-weight:700;color:#ffffff;">{title} ({len(section_rows)})</td></tr>
        <tr><td style="padding:0;">
          <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
            {rows_html}
          </table>
        </td></tr>"""

    body_sections = (
        section("BUY", buys) + section("SELL", sells) + section("HOLD", holds) + section("Not Yet Analyzed", unrated)
    )

    unsubscribe_html = ""
    if unsubscribe_token:
        unsubscribe_html = f"""
        <p style="color:#666666;font-size:11px;margin-top:24px;">
          <a href="{API_BASE_UNSUBSCRIBE_URL}/{unsubscribe_token}" style="color:#666666;">Unsubscribe</a>
        </p>"""

    return f"""<!doctype html>
<html>
<body style="margin:0;padding:0;background:#0a0a0f;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;margin:0 auto;background:#13131f;">
    <tr>
      <td style="padding:24px 12px 8px;">
        <h1 style="color:#ffffff;font-size:20px;margin:0;">StockPulse Daily Signals</h1>
        <p style="color:#888888;font-size:12px;margin:4px 0 0;">{as_of.strftime("%A, %B %-d, %Y — %-I:%M %p")}</p>
      </td>
    </tr>
    {body_sections}
    <tr>
      <td style="padding:20px 12px;">
        <p style="color:#666666;font-size:11px;line-height:1.5;margin:0;">
          For informational purposes only. Not financial advice. AI-generated signals can be wrong.
        </p>
        {unsubscribe_html}
      </td>
    </tr>
  </table>
</body>
</html>"""
