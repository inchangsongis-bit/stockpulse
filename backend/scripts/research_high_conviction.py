"""
What do the model's highest-confidence calls actually look like, and can
that be turned into a gate — only predict when those conditions hold?

Method discipline matters more than usual here. With 2.1M test rows and
five features there are endless conditions to slice on, and some will look
excellent by chance alone. So everything is derived from the TRAINING
half and evaluated exactly once on the held-out half:

  A. CHARACTERISTICS  What distinguishes top-1% confidence calls: feature
     values, direction bias, time of day, realized volatility, and how the
     call relates to recent price action.

  B. MOMENTUM OR REVERSION?  Does the model call WITH the recent move or
     AGAINST it when it's confident? This is the single most informative
     thing about what it has learned.

  C. ACCURACY BY FEATURE RANGE  Measured on TRAIN only, to generate
     candidate rules without touching the test set.

  D. RULE VALIDATION  The rules from (C) applied once to TEST. A rule that
     holds up here is real; one that collapses was overfitting to train.

  E. RISK-COVERAGE CURVE  The fundamental selective-prediction tradeoff:
     as the model is allowed to abstain more, how does accuracy on what
     remains actually rise? This bounds what any gate can achieve.

Run: cd backend && source venv/bin/activate && python scripts/research_high_conviction.py
"""

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.forecast_features import FEATURE_COLUMNS, HORIZON_MINUTES, build_features  # noqa: E402

warnings.filterwarnings("ignore")
DB_PATH = Path(__file__).resolve().parent.parent / "stockpulse.db"
TEST_FRAC = 0.2
# The reference series pulled for the intermarket test aren't watchlist
# instruments and shouldn't be modelled as if they were.
EXCLUDE = {"VXX", "TLT", "UUP", "GLD"}


def load_all():
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        df = pd.read_sql_query(
            "SELECT ticker, timestamp, open, high, low, close, volume FROM ohlcv "
            "WHERE interval='minute' ORDER BY ticker, timestamp",
            conn, parse_dates=["timestamp"],
        )
    return {t: g.reset_index(drop=True) for t, g in df.groupby("ticker", sort=False)
            if t not in EXCLUDE}


def assemble(bars_by_ticker):
    tr, te = [], []
    for ticker, bars in bars_by_ticker.items():
        if len(bars) < 500:
            continue
        f = build_features(bars)
        close, ts = bars["close"], pd.to_datetime(bars["timestamp"])
        fwd = (close.shift(-HORIZON_MINUTES) - close) / close
        gap = (ts.shift(-HORIZON_MINUTES) - ts).dt.total_seconds() / 60
        fwd[gap != HORIZON_MINUTES] = np.nan
        f["fwd_bps"] = fwd * 10000
        f["label"] = (fwd > 0).astype(float)
        f.loc[fwd.isna(), "label"] = np.nan
        f["ticker"] = ticker

        keep = f[FEATURE_COLUMNS].notna().all(axis=1) & f["label"].notna()
        d = f.loc[keep, FEATURE_COLUMNS + ["fwd_bps", "label", "ticker", "timestamp"]].reset_index(drop=True)
        cut = int(len(d) * (1 - TEST_FRAC))
        tr.append(d.iloc[:cut]); te.append(d.iloc[cut:])
    return pd.concat(tr, ignore_index=True), pd.concat(te, ignore_index=True)


def annotate(df, model):
    p = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    out = df.copy()
    out["proba"] = p
    out["conf"] = np.abs(p - 0.5)
    out["pred"] = (p >= 0.5).astype(int)
    out["correct"] = (out["pred"] == out["label"]).astype(int)
    out["minute_of_day"] = pd.to_datetime(out["timestamp"]).dt.hour * 60 + \
                           pd.to_datetime(out["timestamp"]).dt.minute
    return out


def main():
    print("Loading pooled minute data...")
    bars = load_all()
    train, test = assemble(bars)
    print(f"  {len(bars)} tickers | train={len(train):,}  test={len(test):,}\n")

    model = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    model.fit(train[FEATURE_COLUMNS], train["label"])
    tr = annotate(train, model)
    te = annotate(test, model)

    # Confidence cut defined on TRAIN, applied unchanged to TEST.
    cut99 = np.percentile(tr["conf"], 99)
    tr_hi, te_hi = tr[tr.conf >= cut99], te[te.conf >= cut99]

    print("=" * 100)
    print("A — What do the top-1% confidence calls look like? (feature means)")
    print("=" * 100)
    print(f"  {'':<18s}{'all calls':>14s}{'top-1% conf':>14s}{'ratio':>10s}")
    for c in FEATURE_COLUMNS:
        a, b = tr[c].mean(), tr_hi[c].mean()
        ratio = b / a if a else float("nan")
        print(f"  {c:<18s}{a:>14.6f}{b:>14.6f}{ratio:>10.2f}x")
    print(f"  {'|ret_5| (abs)':<18s}{tr['ret_5'].abs().mean():>14.6f}"
          f"{tr_hi['ret_5'].abs().mean():>14.6f}"
          f"{tr_hi['ret_5'].abs().mean() / tr['ret_5'].abs().mean():>10.2f}x")
    print(f"\n  {'|fwd move| bps':<18s}{tr['fwd_bps'].abs().mean():>14.2f}"
          f"{tr_hi['fwd_bps'].abs().mean():>14.2f}")
    print(f"  {'calls that are UP':<18s}{tr['pred'].mean():>14.2%}{tr_hi['pred'].mean():>14.2%}")

    print("\n" + "=" * 100)
    print("B — Momentum or mean reversion? (does it call WITH or AGAINST recent move)")
    print("=" * 100)
    for name, d in (("all calls", tr), ("top-1% conf", tr_hi)):
        agree = ((d["pred"] == 1) == (d["ret_5"] > 0)).mean()
        print(f"  {name:<14s}: call agrees with the last 5-min move {agree:.2%} of the time")
    print("\n  Below 50% means the model is predominantly a MEAN-REVERSION model —")
    print("  it bets against the recent move. Above 50% means momentum.")

    print("\n" + "=" * 100)
    print("C — Accuracy by feature range, measured on TRAIN ONLY (rule generation)")
    print("=" * 100)
    candidates = []
    for c in FEATURE_COLUMNS + ["minute_of_day"]:
        try:
            q = pd.qcut(tr_hi[c], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        accs = tr_hi.groupby(q)["correct"].agg(["mean", "count"])
        best_bin = accs["mean"].idxmax()
        lo = tr_hi[c][q == best_bin].min()
        hi = tr_hi[c][q == best_bin].max()
        print(f"  {c:<16s} best quintile acc={accs['mean'].max():.4f} "
              f"(n={int(accs['count'][best_bin]):,})  range [{lo:.6g}, {hi:.6g}]")
        candidates.append((c, lo, hi, accs["mean"].max()))

    print("\n" + "=" * 100)
    print("D — Those rules applied ONCE to the held-out TEST set")
    print("=" * 100)
    base_hi = te_hi["correct"].mean()
    print(f"  top-1% confidence, no extra rule : n={len(te_hi):>7,d}  acc={base_hi:.4f}")
    print()
    for c, lo, hi, tr_acc in candidates:
        sel = te_hi[(te_hi[c] >= lo) & (te_hi[c] <= hi)]
        if len(sel) < 200:
            print(f"  + {c:<16s} test n too small ({len(sel)})")
            continue
        print(f"  + {c:<16s} train acc={tr_acc:.4f} -> TEST acc={sel['correct'].mean():.4f}  "
              f"(n={len(sel):,}, {len(sel) / len(te_hi):.1%} of calls)  "
              f"{'HOLDS' if sel['correct'].mean() > base_hi else 'does not hold'}")

    print("\n" + "=" * 100)
    print("E — Risk-coverage curve: how much does abstaining actually buy?")
    print("=" * 100)
    print(f"  {'coverage':>10s}{'n':>12s}{'accuracy':>11s}{'avg |move| bps':>17s}")
    for pct in (100, 50, 20, 10, 5, 1, 0.5, 0.1):
        thresh = np.percentile(te["conf"], 100 - pct)
        sel = te[te.conf >= thresh]
        print(f"  {pct:>9.1f}%{len(sel):>12,d}{sel['correct'].mean():>11.4f}"
              f"{sel['fwd_bps'].abs().mean():>17.2f}")

    print("\n  Gross P&L per trade = avg|move| x (2*accuracy - 1); a 1-2bps round trip")
    print("  is the bar it has to clear.")
    print(f"\n  {'coverage':>10s}{'gross bps':>12s}{'net @1bps':>12s}{'net @2bps':>12s}")
    for pct in (100, 10, 1, 0.5, 0.1):
        thresh = np.percentile(te["conf"], 100 - pct)
        sel = te[te.conf >= thresh]
        gross = sel["fwd_bps"].abs().mean() * (2 * sel["correct"].mean() - 1)
        print(f"  {pct:>9.1f}%{gross:>12.3f}{gross - 1:>12.3f}{gross - 2:>12.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
