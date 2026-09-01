"""
Feature engineering for the short-term (5-minute-ahead) direction
forecast. Operates on minute-bar OHLCV — a much shorter horizon than the
daily-bar technical profile in indicators.py, so this reuses that
module's indicator functions but with windows sized for 1-minute bars
instead of daily ones.

Deliberately technical/volume-only — "purely based on the up/down graph,
volume and previous patterns," per the feature request. There's no
historical per-minute sentiment series to train on (only the *current*
composite sentiment is ever known), so news is blended in only at
inference time as a small adjustment on top of this model's own
probability (see analysis/forecast.py), never as a trained feature here.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from analysis.indicators import rsi, sma, volume_sma_ratio

FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "vol_ratio",
    "rsi_7",
    "price_vs_sma5",
    "volatility_10",
]

HORIZON_MINUTES = 5

# Rolling windows need this many warm-up bars before every feature is
# defined (vol_ratio's 20-period SMA is the longest).
MIN_WARMUP_BARS = 20


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: minute bars sorted ascending by timestamp, with columns
    [timestamp, open, high, low, close, volume]. Returns a copy with the
    FEATURE_COLUMNS appended — the first MIN_WARMUP_BARS-ish rows will
    have NaN features until their rolling windows fill up; callers doing
    training should dropna(subset=FEATURE_COLUMNS), and callers doing a
    single live prediction should just take the last row (which is
    guaranteed fully warmed up as long as at least MIN_WARMUP_BARS rows
    were passed in).
    """
    out = df.copy()
    close = out["close"]
    sma5 = sma(close, 5)

    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)
    out["vol_ratio"] = volume_sma_ratio(out["volume"], period=20)
    out["rsi_7"] = rsi(close, period=7)
    out["price_vs_sma5"] = (close - sma5) / sma5
    out["volatility_10"] = close.rolling(window=10).std() / close

    return out


def build_labels(df: pd.DataFrame, horizon: int = HORIZON_MINUTES) -> pd.Series:
    """
    Binary label: 1.0 if close `horizon` bars ahead is strictly higher
    than the current close, 0.0 if lower-or-equal, NaN if that future bar
    doesn't exist yet OR isn't actually `horizon` minutes later in
    wall-clock time — a session boundary (end of day, weekend, a data
    gap) sits between them, so "5 bars ahead" wouldn't mean "5 minutes
    ahead" there.
    """
    close = df["close"]
    ts = pd.to_datetime(df["timestamp"])
    future_close = close.shift(-horizon)
    future_ts = ts.shift(-horizon)
    gap_minutes = (future_ts - ts).dt.total_seconds() / 60

    label = (future_close > close).astype(float)
    label[gap_minutes != horizon] = np.nan
    return label


def build_training_set(bars_by_ticker: dict) -> Tuple[pd.DataFrame, pd.Series]:
    """
    bars_by_ticker: {ticker: DataFrame of minute bars for that ticker,
    ascending by timestamp}. Builds features + labels per ticker
    (features/labels must never be computed across a ticker boundary)
    then pools everything into one training set.
    """
    feature_frames = []
    label_series = []

    for ticker, bars in bars_by_ticker.items():
        if len(bars) < MIN_WARMUP_BARS + HORIZON_MINUTES:
            continue
        featured = build_features(bars)
        labels = build_labels(bars)
        feature_frames.append(featured[FEATURE_COLUMNS])
        label_series.append(labels)

    if not feature_frames:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=float)

    X = pd.concat(feature_frames, ignore_index=True)
    y = pd.concat(label_series, ignore_index=True)

    valid = X.notna().all(axis=1) & y.notna()
    return X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)


def build_chronological_split(
    bars_by_ticker: dict, test_frac: float = 0.2
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits each ticker's OWN timeline test_frac/rest (not the pooled,
    ticker-then-time-concatenated array), then concatenates all train
    pieces and all test pieces. A single cut on the pooled array — the
    naive approach — mostly separates tickers (whichever land after the
    cut point alphabetically) rather than separating time, so its "test
    accuracy" partly measures generalization to unseen tickers instead of
    the future. This is the split that should actually back any reported
    accuracy number.
    """
    train_X, train_y, test_X, test_y = [], [], [], []

    for ticker, bars in bars_by_ticker.items():
        if len(bars) < MIN_WARMUP_BARS + HORIZON_MINUTES:
            continue
        featured = build_features(bars)
        labels = build_labels(bars)
        valid = featured[FEATURE_COLUMNS].notna().all(axis=1) & labels.notna()
        X = featured.loc[valid, FEATURE_COLUMNS].reset_index(drop=True)
        y = labels.loc[valid].reset_index(drop=True)
        if len(X) < 20:
            continue
        split = int(len(X) * (1 - test_frac))
        train_X.append(X.iloc[:split])
        train_y.append(y.iloc[:split])
        test_X.append(X.iloc[split:])
        test_y.append(y.iloc[split:])

    if not train_X:
        empty = pd.DataFrame(columns=FEATURE_COLUMNS)
        return empty, pd.Series(dtype=float), empty, pd.Series(dtype=float)

    return (
        pd.concat(train_X, ignore_index=True), pd.concat(train_y, ignore_index=True),
        pd.concat(test_X, ignore_index=True), pd.concat(test_y, ignore_index=True),
    )
