"""
Re-runs the multi-day-context test from research_spy.py after fixing a
lookahead bug, and isolates which feature was responsible.

The first run reported acc=0.5529 / AUC=0.5625 for "+ multi-day context"
against a 0.5049 / 0.5098 baseline — a fivefold jump in edge, which for a
5-minute equity direction call is not a plausible result. It came from
this line:

    prev_close = df.groupby(date)["close"].transform("last").shift(1)

transform("last") broadcasts each day's FINAL close across every row of
that same day, and .shift(1) then moves it by one ROW rather than one
day. So every intraday bar was reading its own day's closing price.
dist_prev_close became "how far is the price from where it will close
today", which of course predicts direction.

This script reports the buggy and corrected versions side by side, then
each remaining feature on its own, so the corrected contribution of the
genuinely-known-at-time-t features is visible.

Run: cd backend && source venv/bin/activate && python scripts/research_spy_multiday.py
"""

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.forecast_features import FEATURE_COLUMNS, HORIZON_MINUTES, build_features  # noqa: E402

warnings.filterwarnings("ignore")
DB_PATH = Path(__file__).resolve().parent.parent / "stockpulse.db"
TEST_FRAC = 0.2


def build(ticker="SPY"):
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        df = pd.read_sql_query(
            "SELECT timestamp, open, high, low, close, volume FROM ohlcv "
            "WHERE interval='minute' AND ticker=? ORDER BY timestamp",
            conn, params=(ticker,), parse_dates=["timestamp"],
        )
    f = build_features(df)
    close, ts = df["close"], pd.to_datetime(df["timestamp"])
    fwd = (close.shift(-HORIZON_MINUTES) - close) / close
    gap = (ts.shift(-HORIZON_MINUTES) - ts).dt.total_seconds() / 60
    fwd[gap != HORIZON_MINUTES] = np.nan
    f["label"] = (fwd > 0).astype(float)
    f.loc[fwd.isna(), "label"] = np.nan

    d = ts.dt.date
    day_open = f.groupby(d)["open"].transform("first")
    day_high = f.groupby(d)["high"].transform("cummax")
    day_low = f.groupby(d)["low"].transform("cummin")

    f["gap_from_day_open"] = (f["close"] - day_open) / day_open
    f["pos_in_day_range"] = (f["close"] - day_low) / (day_high - day_low).replace(0, np.nan)
    f["ret_60"] = f["close"].pct_change(60)
    f["ret_390"] = f["close"].pct_change(390)

    # The bug, kept deliberately so the two can be compared directly.
    buggy_prev = f.groupby(d)["close"].transform("last").shift(1)
    f["dist_prev_close_BUGGY"] = (f["close"] - buggy_prev) / buggy_prev

    # The fix: aggregate to daily, shift at the daily level, map back.
    daily_close = f.groupby(d)["close"].last()
    correct_prev = d.map(daily_close.shift(1))
    f["dist_prev_close"] = (f["close"] - correct_prev) / correct_prev
    return f


def run(f, cols, name):
    valid = f[cols].notna().all(axis=1) & f["label"].notna()
    X = f.loc[valid, cols].reset_index(drop=True)
    y = f.loc[valid, "label"].reset_index(drop=True)
    cut = int(len(X) * (1 - TEST_FRAC))
    Xtr, ytr, Xte, yte = X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    acc = accuracy_score(yte, (p >= 0.5).astype(int))
    try:
        auc = roc_auc_score(yte, p)
    except ValueError:
        auc = float("nan")
    maj = max(yte.mean(), 1 - yte.mean())
    print(f"  {name:<48s} n={len(Xte):>6,d}  acc={acc:.4f} (maj {maj:.4f}, "
          f"edge {acc - maj:+.4f})  AUC={auc:.4f}")
    return auc


def main():
    print("Building SPY features...")
    f = build()
    print(f"  {len(f):,} bars\n")

    print("=" * 100)
    print("The bug, isolated")
    print("=" * 100)
    run(f, FEATURE_COLUMNS, "baseline (no multi-day features)")
    run(f, FEATURE_COLUMNS + ["dist_prev_close_BUGGY"], "+ dist_prev_close  [LOOKAHEAD BUG]")
    run(f, FEATURE_COLUMNS + ["dist_prev_close"], "+ dist_prev_close  [corrected]")

    print("\n" + "=" * 100)
    print("Each genuinely-known-at-time-t feature on its own")
    print("=" * 100)
    for col in ["gap_from_day_open", "pos_in_day_range", "ret_60", "ret_390"]:
        run(f, FEATURE_COLUMNS + [col], f"+ {col}")

    print("\n" + "=" * 100)
    print("All corrected multi-day features together")
    print("=" * 100)
    clean = ["gap_from_day_open", "pos_in_day_range", "dist_prev_close", "ret_60", "ret_390"]
    run(f, FEATURE_COLUMNS + clean, "+ multi-day context (corrected)")
    print()
    run(f, FEATURE_COLUMNS + clean + ["dist_prev_close_BUGGY"],
        "+ multi-day context (with the bug, for reference)")

    print("\nDone.")


if __name__ == "__main__":
    main()
