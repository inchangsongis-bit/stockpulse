from datetime import datetime

from services.email_templates import render_daily_digest_html


def make_row(ticker, price=100.0, action=None, confidence=None):
    signal = {"action": action, "confidence": confidence} if action else None
    return {"ticker": ticker, "price": price, "signal": signal}


def test_digest_groups_rows_by_action():
    rows = [
        make_row("AAPL", action="BUY", confidence=60),
        make_row("MSFT", action="SELL", confidence=40),
        make_row("SPY", action="HOLD", confidence=20),
        make_row("HSBC", price=None),  # unrated
    ]

    html = render_daily_digest_html(rows, as_of=datetime(2026, 8, 30, 6, 25))

    assert "BUY (1)" in html
    assert "SELL (1)" in html
    assert "HOLD (1)" in html
    assert "Not Yet Analyzed (1)" in html
    assert "AAPL" in html
    assert "MSFT" in html
    assert "SPY" in html
    assert "HSBC" in html
    assert "—" in html  # HSBC's null price


def test_digest_omits_empty_sections():
    rows = [make_row("AAPL", action="BUY", confidence=60)]

    html = render_daily_digest_html(rows, as_of=datetime(2026, 8, 30))

    assert "BUY (1)" in html
    assert "SELL" not in html
    assert "HOLD" not in html
    assert "Not Yet Analyzed" not in html


def test_digest_includes_unsubscribe_link_only_when_token_given():
    rows = [make_row("AAPL", action="BUY", confidence=60)]

    with_token = render_daily_digest_html(rows, as_of=datetime(2026, 8, 30), unsubscribe_token="tok123")
    without_token = render_daily_digest_html(rows, as_of=datetime(2026, 8, 30))

    assert "unsubscribe/tok123" in with_token
    assert "Unsubscribe" not in without_token


def test_digest_includes_disclaimer():
    html = render_daily_digest_html([make_row("AAPL", action="BUY", confidence=50)], as_of=datetime(2026, 8, 30))
    assert "Not financial advice" in html
