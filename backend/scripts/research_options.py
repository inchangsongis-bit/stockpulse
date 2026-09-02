"""
Does options data create an edge where the price-only model has none?

The price/volume model's apparent skill turned out to live entirely in
extended-hours illiquidity: 61.6% accuracy at the top 0.1% of confidence
across all hours, but 52.9% inside regular hours, with net P&L negative
at every coverage level there. So the bar for options features is not
"does AUC go up" in general — it is specifically:

    does anything improve during REGULAR TRADING HOURS, where a trade
    could actually be executed?

Options markets are closed outside those hours anyway, so this is the
only window where the question even makes sense.

Tested here, all on SPY, all evaluated on a chronological holdout:

  1. Price-only baseline, restricted to regular hours (the number to beat)
  2. + put/call volume ratio
  3. + implied volatility and skew (recovered by inverting Black-Scholes,
     since Polygon's greeks endpoint is 403 on this tier)
  4. + all options features together
  5. Risk-coverage curve with net P&L, to see whether selective
     prediction becomes viable when it wasn't before

Run: cd backend && source venv/bin/activate && python scripts/research_options.py
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
from analysis.options_features import build_options_features  # noqa: E402

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
MAIN_DB = BASE / "stockpulse.db"
PILOT_DB = BASE / "options_pilot.db"
RTH_START_MIN, RTH_END_MIN = 390, 780
TEST_FRAC = 0.2


def load_spot():
    with sqlite3.connect(f"file:{MAIN_DB}?mode=ro", uri=True) as c:
        return pd.read_sql_query(
            "SELECT timestamp, open, high, low, close, volume FROM ohlcv "
            "WHERE interval='minute' AND ticker='SPY' ORDER BY timestamp",
            c, parse_dates=["timestamp"],
        )


def load_options():
    with sqlite3.connect(f"file:{PILOT_DB}?mode=ro", uri=True) as c:
        return pd.read_sql_query(
            "SELECT contract, expiration, strike, opt_type, timestamp, close, volume "
            "FROM option_bars ORDER BY timestamp",
            c, parse_dates=["timestamp"],
        )


def evaluate(df, cols, name, baseline_auc=None):
    valid = df[cols].notna().all(axis=1) & df["label"].notna()
    d = df.loc[valid].reset_index(drop=True)
    if len(d) < 500:
        print(f"  {name:<44s} only {len(d)} usable rows — skipped")
        return None, None
    cut = int(len(d) * (1 - TEST_FRAC))
    tr, te = d.iloc[:cut], d.iloc[cut:]
    if tr["label"].nunique() < 2 or te["label"].nunique() < 2:
        print(f"  {name:<44s} single-class split — skipped")
        return None, None

    m = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    m.fit(tr[cols], tr["label"])
    p = m.predict_proba(te[cols])[:, 1]
    acc = accuracy_score(te["label"], (p >= 0.5).astype(int))
    auc = roc_auc_score(te["label"], p)
    maj = max(te["label"].mean(), 1 - te["label"].mean())
    delta = f"  ({auc - baseline_auc:+.4f})" if baseline_auc is not None else ""
    print(f"  {name:<44s} n={len(te):>6,d}  acc={acc:.4f} (maj {maj:.4f})  AUC={auc:.4f}{delta}")
    return auc, (te, p)


def main():
    if not PILOT_DB.exists():
        print("options_pilot.db not found — run scripts/fetch_options_pilot.py first.")
        return

    print("Loading SPY and the options pilot dataset...")
    spot = load_spot()
    opts = load_options()
    print(f"  SPY: {len(spot):,} minute bars")
    print(f"  options: {len(opts):,} bars across {opts['contract'].nunique()} contracts, "
          f"{opts['timestamp'].min().date()} -> {opts['timestamp'].max().date()}")

    print("Computing options features (Black-Scholes inversion for IV)...")
    of = build_options_features(opts, spot)
    print(f"  {len(of):,} minutes with options features\n")

    feat = build_features(spot)
    close, ts = spot["close"], spot["timestamp"]
    fwd = (close.shift(-HORIZON_MINUTES) - close) / close
    gap = (ts.shift(-HORIZON_MINUTES) - ts).dt.total_seconds() / 60
    fwd[gap != HORIZON_MINUTES] = np.nan
    feat["fwd_bps"] = fwd * 10000
    feat["label"] = (fwd > 0).astype(float)
    feat.loc[fwd.isna(), "label"] = np.nan

    df = feat.merge(of, on="timestamp", how="inner")
    mod = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df = df[(mod >= RTH_START_MIN) & (mod <= RTH_END_MIN)].reset_index(drop=True)
    print(f"Overlapping regular-hours minutes with both price and options data: {len(df):,}")
    print(f"  span: {df['timestamp'].min()} -> {df['timestamp'].max()}\n")

    print("=" * 100)
    print("Does anything help during REGULAR HOURS? (the only executable window)")
    print("=" * 100)
    base_auc, base_out = evaluate(df, list(FEATURE_COLUMNS), "1. price-only baseline (RTH)")
    if base_auc is None:
        print("\nNot enough overlapping data yet — let the fetcher finish and re-run.")
        return

    evaluate(df, list(FEATURE_COLUMNS) + ["put_call_volume_ratio", "options_volume_ratio"],
             "2. + put/call ratio, options volume", base_auc)
    evaluate(df, list(FEATURE_COLUMNS) + ["atm_iv", "iv_skew"],
             "3. + implied volatility, skew", base_auc)
    all_auc, all_out = evaluate(
        df, list(FEATURE_COLUMNS) + ["put_call_volume_ratio", "options_volume_ratio",
                                     "atm_iv", "iv_skew"],
        "4. + all options features", base_auc)

    print("\n" + "=" * 100)
    print("Options features on their OWN (no price features)")
    print("=" * 100)
    evaluate(df, ["put_call_volume_ratio", "options_volume_ratio", "atm_iv", "iv_skew"],
             "5. options features only", base_auc)

    if all_out is None:
        print("\nDone.")
        return

    print("\n" + "=" * 100)
    print("Risk-coverage with net P&L — does selective prediction become viable?")
    print("=" * 100)
    te, p = all_out
    conf = np.abs(p - 0.5)
    correct = ((p >= 0.5).astype(int) == te["label"].to_numpy()).astype(int)
    move = te["fwd_bps"].abs().to_numpy()
    print(f"  {'coverage':>10s}{'n':>9s}{'accuracy':>10s}{'avg|move|':>11s}"
          f"{'gross bps':>11s}{'net @1bps':>11s}{'net @2bps':>11s}")
    for pct in (100, 25, 10, 5, 1):
        if len(conf) * pct / 100 < 30:
            continue
        thresh = np.percentile(conf, 100 - pct)
        sel = conf >= thresh
        acc = correct[sel].mean()
        mv = move[sel].mean()
        gross = mv * (2 * acc - 1)
        print(f"  {pct:>9d}%{sel.sum():>9,d}{acc:>10.4f}{mv:>11.2f}"
              f"{gross:>11.3f}{gross - 1:>11.3f}{gross - 2:>11.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
