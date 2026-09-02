"""
One-off research script (not part of the app) comparing candidate
improvements to the 5-minute forecast model against the current
production baseline, on our own real minute-bar data:

  1. Baseline: current feature set (forecast_features.py) + raw
     sign-of-return labels + HistGradientBoostingClassifier + a single
     80/20 split.
  2. + Proper per-ticker chronological split (the original 80/20 split
     concatenates tickers alphabetically then cuts once — that's mostly
     a ticker-generalization test, not a time-based one).
  3. + Expanded feature set (more technical/volume features, still
     OHLCV-only).
  4. + Dead-zone labeling (drop near-flat moves the model can't
     realistically call; report coverage alongside accuracy).
  5. LightGBM vs HistGradientBoostingClassifier on the same data.

Findings (see PR description / commit message for the full writeup):
none of these moved accuracy meaningfully past ~52% — consistent with
the efficient-market/random-walk literature for this horizon. The
per-ticker chronological split (item 2) was the one legitimate fix and
is now what scripts/train_forecast_model.py actually uses.

lightgbm isn't a project dependency (not worth adding for a change that
didn't help) — install it separately to re-run this:
    pip install lightgbm
    brew install libomp   # macOS: lightgbm's compiled binary needs this

Run: cd backend && source venv/bin/activate && python scripts/research_forecast_improvements.py
"""

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.forecast_features import (  # noqa: E402
    FEATURE_COLUMNS as BASELINE_FEATURES,
    HORIZON_MINUTES,
    build_features as build_baseline_features,
    build_labels,
)
from analysis.indicators import macd, bollinger_bands, atr, sma  # noqa: E402
from database import async_session  # noqa: E402
from models import OHLCV  # noqa: E402

EXPANDED_FEATURES = BASELINE_FEATURES + [
    "ret_10",
    "macd_hist_fast",
    "bb_bandwidth_10",
    "atr_ratio_10",
    "price_vs_sma10",
    "ret_1_lag1",
    "ret_1_lag2",
    "tod_sin",
    "tod_cos",
]

DEAD_ZONE_THRESHOLD = 0.0005  # 0.05% — roughly a noise floor for 5-min moves


def build_expanded_features(df: pd.DataFrame) -> pd.DataFrame:
    out = build_baseline_features(df)
    close = out["close"]

    out["ret_10"] = close.pct_change(10)
    _, _, macd_hist = macd(close, fast=6, slow=13, signal=4)
    out["macd_hist_fast"] = macd_hist / close
    _, _, _, bb_bw = bollinger_bands(close, period=10)
    out["bb_bandwidth_10"] = bb_bw
    out["atr_ratio_10"] = atr(out, period=10) / close
    sma10 = sma(close, 10)
    out["price_vs_sma10"] = (close - sma10) / sma10
    out["ret_1_lag1"] = out["ret_1"].shift(1)
    out["ret_1_lag2"] = out["ret_1"].shift(2)

    minutes_since_midnight = pd.to_datetime(out["timestamp"]).dt.hour * 60 + pd.to_datetime(out["timestamp"]).dt.minute
    frac_of_day = (minutes_since_midnight % 1440) / 1440
    out["tod_sin"] = np.sin(2 * np.pi * frac_of_day)
    out["tod_cos"] = np.cos(2 * np.pi * frac_of_day)

    return out


def build_labels_dead_zone(df: pd.DataFrame, horizon: int = HORIZON_MINUTES, threshold: float = DEAD_ZONE_THRESHOLD) -> pd.Series:
    close = df["close"]
    ts = pd.to_datetime(df["timestamp"])
    future_close = close.shift(-horizon)
    future_ts = ts.shift(-horizon)
    gap_minutes = (future_ts - ts).dt.total_seconds() / 60
    ret = (future_close - close) / close

    label = pd.Series(np.nan, index=df.index)
    label[ret > threshold] = 1.0
    label[ret < -threshold] = 0.0
    label[gap_minutes != horizon] = np.nan
    return label


async def load_minute_bars_by_ticker() -> dict:
    async with async_session() as db:
        result = await db.execute(
            select(OHLCV).where(OHLCV.interval == "minute").order_by(OHLCV.ticker, OHLCV.timestamp.asc())
        )
        rows = result.scalars().all()

    bars_by_ticker = defaultdict(list)
    for r in rows:
        bars_by_ticker[r.ticker].append({
            "timestamp": r.timestamp, "open": r.open, "high": r.high,
            "low": r.low, "close": r.close, "volume": r.volume,
        })
    return {t: pd.DataFrame(bars) for t, bars in bars_by_ticker.items()}


def per_ticker_chronological_split(
    bars_by_ticker: dict, feature_fn, label_fn, feature_columns: list, test_frac: float = 0.2
):
    """Splits each ticker's own timeline 80/20 (not the pooled array), then
    concatenates all train pieces and all test pieces — a real time-based
    holdout instead of an accidental ticker-generalization test."""
    train_X, train_y, test_X, test_y = [], [], [], []

    for ticker, bars in bars_by_ticker.items():
        if len(bars) < 50:
            continue
        featured = feature_fn(bars)
        labels = label_fn(bars)
        valid = featured[feature_columns].notna().all(axis=1) & labels.notna()
        X = featured.loc[valid, feature_columns].reset_index(drop=True)
        y = labels.loc[valid].reset_index(drop=True)
        if len(X) < 20:
            continue
        split = int(len(X) * (1 - test_frac))
        train_X.append(X.iloc[:split])
        train_y.append(y.iloc[:split])
        test_X.append(X.iloc[split:])
        test_y.append(y.iloc[split:])

    return (
        pd.concat(train_X, ignore_index=True), pd.concat(train_y, ignore_index=True),
        pd.concat(test_X, ignore_index=True), pd.concat(test_y, ignore_index=True),
    )


def evaluate(model, X_train, y_train, X_test, y_test, label: str, coverage: float = 1.0):
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")
    print(f"{label:55s}  n_train={len(X_train):>7d}  n_test={len(X_test):>6d}  "
          f"coverage={coverage:5.1%}  accuracy={acc:.4f}  AUC={auc:.4f}")
    return acc, auc


def main():
    print("Loading minute bars...")
    bars_by_ticker = asyncio.run(load_minute_bars_by_ticker())
    total_bars = sum(len(b) for b in bars_by_ticker.values())
    print(f"  {len(bars_by_ticker)} tickers, {total_bars} total bars\n")

    print("=" * 100)
    print("1. BASELINE (current production setup: original features, raw labels, alphabetical-cut split)")
    print("=" * 100)
    from analysis.forecast_features import build_training_set
    X, y = build_training_set(bars_by_ticker)
    split = int(len(X) * 0.8)
    evaluate(
        HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42),
        X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:],
        "Baseline (original split, as currently in production)",
    )

    print("\n" + "=" * 100)
    print("2. Same features/labels, but a REAL per-ticker chronological split")
    print("=" * 100)
    Xtr, ytr, Xte, yte = per_ticker_chronological_split(
        bars_by_ticker, build_baseline_features, build_labels, BASELINE_FEATURES
    )
    evaluate(
        HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42),
        Xtr, ytr, Xte, yte, "Original features, proper time-based split",
    )

    print("\n" + "=" * 100)
    print("3. EXPANDED features (still OHLCV-only), proper time-based split")
    print("=" * 100)
    Xtr, ytr, Xte, yte = per_ticker_chronological_split(
        bars_by_ticker, build_expanded_features, build_labels, EXPANDED_FEATURES
    )
    evaluate(
        HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42),
        Xtr, ytr, Xte, yte, "Expanded features, proper time-based split",
    )
    hgb_expanded = (Xtr, ytr, Xte, yte)

    print("\n" + "=" * 100)
    print("4. EXPANDED features + LightGBM instead of HistGradientBoosting")
    print("=" * 100)
    Xtr, ytr, Xte, yte = hgb_expanded
    evaluate(
        LGBMClassifier(max_depth=4, n_estimators=200, random_state=42, verbose=-1),
        Xtr, ytr, Xte, yte, "Expanded features + LightGBM",
    )

    print("\n" + "=" * 100)
    print("5. EXPANDED features + DEAD-ZONE labeling (drop near-flat moves)")
    print("=" * 100)
    Xtr, ytr, Xte, yte = per_ticker_chronological_split(
        bars_by_ticker, build_expanded_features, build_labels_dead_zone, EXPANDED_FEATURES
    )
    # Coverage: how much of the original (non-dead-zone) test set this
    # subset represents — a dead-zone model that only "votes" on 40% of
    # moments and abstains the rest is a different product, not directly
    # comparable to a model with 100% coverage.
    _, _, Xte_all, _ = per_ticker_chronological_split(
        bars_by_ticker, build_expanded_features, build_labels, EXPANDED_FEATURES
    )
    coverage = len(Xte) / len(Xte_all) if len(Xte_all) else float("nan")
    evaluate(
        HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42),
        Xtr, ytr, Xte, yte, "Expanded features + dead-zone labels", coverage=coverage,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
