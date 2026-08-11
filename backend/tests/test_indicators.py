from datetime import datetime, timedelta

import pandas as pd
import pytest

from analysis.indicators import compute_all_indicators, rsi, sma


def make_df(closes, start=None):
    start = start or datetime(2024, 1, 1)
    n = len(closes)
    return pd.DataFrame({
        "timestamp": [start + timedelta(days=i) for i in range(n)],
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [1_000_000] * n,
    })


def test_sma_matches_manual_average():
    s = pd.Series([1, 2, 3, 4, 5])
    result = sma(s, 3)
    assert result.iloc[-1] == pytest.approx((3 + 4 + 5) / 3)
    assert pd.isna(result.iloc[0])


def test_rsi_is_100_when_strictly_rising():
    closes = pd.Series([float(i) for i in range(1, 30)])
    result = rsi(closes, period=14)
    assert result.iloc[-1] == pytest.approx(100.0, abs=0.01)


def test_compute_all_indicators_handles_short_series():
    # Fewer than 20 bars — most windows can't fill, should not raise
    df = make_df([100 + i for i in range(10)])
    result = compute_all_indicators(df)

    assert result["volatility_state"] == "unknown"
    assert -1.0 <= result["trend_score"] <= 1.0
    assert -1.0 <= result["momentum_score"] <= 1.0
    assert "current_price" in result["indicators"]


def test_compute_all_indicators_uptrend_has_positive_trend_score():
    # 250 bars of steady uptrend
    closes = [100 + i * 0.5 for i in range(250)]
    df = make_df(closes)
    result = compute_all_indicators(df)

    assert result["trend_score"] > 0
    assert result["indicators"]["current_price"] == pytest.approx(closes[-1], abs=0.01)


def test_compute_all_indicators_downtrend_has_negative_trend_score():
    closes = [200 - i * 0.5 for i in range(250)]
    df = make_df(closes)
    result = compute_all_indicators(df)

    assert result["trend_score"] < 0


def test_support_resistance_bracket_recent_prices():
    closes = [100, 105, 95, 110, 90, 108, 102]
    df = make_df(closes)
    result = compute_all_indicators(df)

    # low = close * 0.99, high = close * 1.01 in make_df
    assert result["support"] == pytest.approx(min(closes) * 0.99, abs=0.01)
    assert result["resistance"] == pytest.approx(max(closes) * 1.01, abs=0.01)
