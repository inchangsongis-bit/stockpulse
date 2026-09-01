from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from analysis.forecast_features import (
    FEATURE_COLUMNS,
    HORIZON_MINUTES,
    build_features,
    build_labels,
    build_training_set,
)


def make_minute_bars(closes, start=None, gap_after_index=None):
    """
    Builds a minute-bar DataFrame from a list of close prices, one bar per
    minute. `gap_after_index`, if given, inserts a 2-hour jump right after
    that index (simulating an overnight/session gap) instead of the usual
    1-minute step.
    """
    start = start or datetime(2026, 1, 2, 9, 30)
    timestamps = []
    t = start
    for i in range(len(closes)):
        timestamps.append(t)
        step = timedelta(hours=2) if i == gap_after_index else timedelta(minutes=1)
        t = t + step
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": [c * 1.001 for c in closes],
        "low": [c * 0.999 for c in closes],
        "close": closes,
        "volume": [1_000_000] * len(closes),
    })


def test_build_features_produces_all_columns_and_warms_up():
    closes = [100 + i * 0.1 for i in range(30)]
    df = make_minute_bars(closes)

    featured = build_features(df)

    for col in FEATURE_COLUMNS:
        assert col in featured.columns
    # Early rows lack full rolling windows
    assert featured["vol_ratio"].iloc[:19].isna().all()
    # By the last row, every rolling window (longest is 20) has filled up
    assert featured[FEATURE_COLUMNS].iloc[-1].notna().all()


def test_build_labels_marks_up_and_down_correctly():
    # Flat, then a clear +5 step up 5 bars later, then flat again
    closes = [100.0] * 10 + [105.0] * 10
    df = make_minute_bars(closes)

    labels = build_labels(df, horizon=HORIZON_MINUTES)

    # Bar at index 5: close=100, close at index 10 (5 ahead) = 105 → up
    assert labels.iloc[5] == 1.0
    # Bar at index 0: close=100, close at index 5 = 100 → not strictly up
    assert labels.iloc[0] == 0.0
    # Last HORIZON_MINUTES bars have no future bar to compare against
    assert labels.iloc[-HORIZON_MINUTES:].isna().all()


def test_build_labels_excludes_session_gaps():
    closes = [100.0 + i for i in range(15)]
    # A 2-hour jump right after index 4 — the bar 5 steps ahead of index 0
    # (index 5) is on the other side of that gap, so it isn't really "5
    # minutes ahead" in wall-clock time.
    df = make_minute_bars(closes, gap_after_index=4)

    labels = build_labels(df, horizon=HORIZON_MINUTES)

    assert np.isnan(labels.iloc[0])


def test_build_training_set_pools_tickers_and_drops_incomplete_rows():
    closes_a = [100 + i * 0.1 for i in range(40)]
    closes_b = [50 - i * 0.05 for i in range(40)]
    too_short = [10.0] * 5  # below MIN_WARMUP_BARS + HORIZON_MINUTES

    bars_by_ticker = {
        "AAA": make_minute_bars(closes_a),
        "BBB": make_minute_bars(closes_b),
        "CCC": make_minute_bars(too_short),
    }

    X, y = build_training_set(bars_by_ticker)

    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y)
    assert len(X) > 0
    assert X.notna().all(axis=None)
    assert y.notna().all()
    assert set(y.unique()) <= {0.0, 1.0}


def test_build_training_set_empty_when_no_ticker_has_enough_bars():
    X, y = build_training_set({"AAA": make_minute_bars([1.0, 2.0, 3.0])})
    assert X.empty
    assert y.empty
