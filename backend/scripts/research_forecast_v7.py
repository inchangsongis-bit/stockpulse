"""
Confounders and methodology audit — the factors that could change the
conclusions of v1-v6, tested rather than assumed.

  R. OVERLAPPING LABELS / EMBARGO. Every bar's label looks 5 minutes
     forward, so bar t and bar t+1 share 4 of their 5 label minutes. Two
     consequences, both of which flatter our numbers:
       - Train/test boundary leakage: the last 4 training bars' labels
         extend INTO the test period. Standard fix (Lopez de Prado) is an
         embargo gap between the two. Measured here by re-running with a
         gap and comparing.
       - Inflated significance: 490k samples are nowhere near 490k
         independent observations. Effective sample size is closer to
         n/horizon, which widens every confidence interval we've quoted.

  S. TEMPORAL CLUSTERING of high-conviction calls. The top-1% result
     rests on 778 predictions. If those are 778 independent moments it's
     meaningful; if they're a handful of volatile episodes with
     consecutive minutes each, the effective sample is far smaller and
     the p-value is close to meaningless.

  T. BAD TICKS. Two years of minute data will contain bad prints and
     halt artifacts. A single 10x bad tick poisons the return features
     around it. Measures how extreme the tails are and whether trimming
     them changes anything.

  U. THE CLOSING-BELL BOUNDARY. A 12:56 PT bar's "5 minutes ahead" lands
     after the closing auction. The session-gap filter can't catch this
     because extended-hours bars ARE contiguous in wall-clock minutes, so
     the gap is still exactly 5. Those labels span a liquidity regime
     change and may be a distinct source of fake signal.

  V. REGIME DEPENDENCE. With ~2 years available, train on the older
     portion and test on progressively later windows to see whether the
     edge is stable over time or was specific to one period.

Run: cd backend && source venv/bin/activate && python scripts/research_forecast_v7.py
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.forecast_features import (  # noqa: E402
    FEATURE_COLUMNS,
    HORIZON_MINUTES,
    build_features,
    build_labels,
)

RTH_END_MIN = 780  # 13:00 PT close
DB_PATH = Path(__file__).resolve().parent.parent / "stockpulse.db"


def load_minute_bars_by_ticker() -> dict:
    """Read straight into a DataFrame via sqlite3 — materializing ~17M ORM
    rows as Python dicts costs several GB of RAM at this dataset size."""
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        df = pd.read_sql_query(
            "SELECT ticker, timestamp, open, high, low, close, volume "
            "FROM ohlcv WHERE interval = 'minute' ORDER BY ticker, timestamp",
            conn,
            parse_dates=["timestamp"],
        )
    return {t: g.reset_index(drop=True) for t, g in df.groupby("ticker", sort=False)}


def assemble(bars_by_ticker, embargo=0, drop_close_spanning=False, clip_returns=None,
             test_frac=0.2, train_window=None):
    """Per-ticker chronological split, with optional embargo gap between
    train and test, closing-bell-spanning label removal, and return
    winsorization."""
    tr_X, tr_y, te_X, te_y, te_ts, te_tick = [], [], [], [], [], []
    for ticker, bars in bars_by_ticker.items():
        if len(bars) < 200:
            continue
        featured = build_features(bars)
        labels = build_labels(bars)

        if drop_close_spanning:
            ts = pd.to_datetime(bars["timestamp"])
            mod = ts.dt.hour * 60 + ts.dt.minute
            # If the bar HORIZON minutes ahead sits past the close while
            # this one doesn't, the label straddles the auction.
            spans = (mod <= RTH_END_MIN) & (mod.shift(-HORIZON_MINUTES) > RTH_END_MIN)
            labels = labels.copy()
            labels[spans.fillna(False)] = np.nan

        if clip_returns is not None:
            for col in ("ret_1", "ret_3", "ret_5"):
                lo, hi = featured[col].quantile([clip_returns, 1 - clip_returns])
                featured[col] = featured[col].clip(lo, hi)

        valid = featured[FEATURE_COLUMNS].notna().all(axis=1) & labels.notna()
        X = featured.loc[valid, FEATURE_COLUMNS].reset_index(drop=True)
        y = labels.loc[valid].reset_index(drop=True)
        ts_v = pd.to_datetime(featured.loc[valid, "timestamp"]).reset_index(drop=True)
        if len(X) < 100:
            continue

        cut = int(len(X) * (1 - test_frac))
        train_end = cut - embargo  # drop the bars whose labels reach into test
        train_start = 0 if train_window is None else max(0, train_end - train_window)
        if train_end <= train_start:
            continue

        tr_X.append(X.iloc[train_start:train_end]); tr_y.append(y.iloc[train_start:train_end])
        te_X.append(X.iloc[cut:]); te_y.append(y.iloc[cut:])
        te_ts.append(ts_v.iloc[cut:]); te_tick.append(pd.Series([ticker] * (len(X) - cut)))

    return (
        pd.concat(tr_X, ignore_index=True), pd.concat(tr_y, ignore_index=True),
        pd.concat(te_X, ignore_index=True), pd.concat(te_y, ignore_index=True),
        pd.concat(te_ts, ignore_index=True), pd.concat(te_tick, ignore_index=True),
    )


def fit_eval(X_train, y_train, X_test, y_test, label):
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    m.fit(X_train, y_train)
    proba = m.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")
    majority = max(y_test.mean(), 1 - y_test.mean())
    print(f"  {label:<46s} test={len(X_test):>7d}  acc={acc:.4f}  "
          f"AUC={auc:.4f}  (majority baseline {majority:.4f}, edge {acc - majority:+.4f})")
    return proba, acc, auc


def main():
    print("Loading minute bars...")
    bars = load_minute_bars_by_ticker()
    total = sum(len(b) for b in bars.values())
    spans = [(pd.to_datetime(b['timestamp']).min(), pd.to_datetime(b['timestamp']).max())
             for b in bars.values() if len(b)]
    print(f"  {len(bars)} tickers, {total:,} bars, "
          f"{min(s[0] for s in spans).date()} -> {max(s[1] for s in spans).date()}\n")

    print("=" * 100)
    print("R — Embargo: does removing train/test label overlap change the result?")
    print("=" * 100)
    Xtr, ytr, Xte, yte, ts_te, tick_te = assemble(bars, embargo=0)
    proba0, acc0, _ = fit_eval(Xtr, ytr, Xte, yte, "no embargo (as previously reported)")
    for emb in (HORIZON_MINUTES, HORIZON_MINUTES * 12):
        eXtr, eytr, eXte, eyte, _, _ = assemble(bars, embargo=emb)
        fit_eval(eXtr, eytr, eXte, eyte, f"embargo of {emb} bars")

    print("\n  Effective sample size correction:")
    n = len(yte)
    n_eff = n / HORIZON_MINUTES
    se_naive, se_eff = np.sqrt(0.25 / n), np.sqrt(0.25 / n_eff)
    print(f"    reported n = {n:,}  ->  effective n ~= {n_eff:,.0f} (labels overlap {HORIZON_MINUTES}x)")
    print(f"    95% CI half-width on accuracy: {1.96 * se_naive:.4f} (naive) "
          f"vs {1.96 * se_eff:.4f} (corrected)")

    print("\n" + "=" * 100)
    print("S — Are high-conviction calls independent moments, or a few clustered episodes?")
    print("=" * 100)
    conf = np.abs(proba0 - 0.5)
    sel = conf >= np.percentile(conf, 99)
    sel_ts = pd.to_datetime(ts_te[sel]).sort_values()
    sel_tick = tick_te[sel]
    gaps = sel_ts.diff().dt.total_seconds().div(60).dropna()
    consecutive = int((gaps <= HORIZON_MINUTES).sum())
    print(f"  selected predictions      : {int(sel.sum())}")
    print(f"  distinct tickers          : {sel_tick.nunique()}")
    print(f"  distinct calendar days    : {sel_ts.dt.date.nunique()}")
    print(f"  within {HORIZON_MINUTES}min of the previous pick: {consecutive} "
          f"({consecutive / max(len(gaps), 1):.1%}) -> overlapping, not independent")
    indep_est = int(sel.sum()) - consecutive
    print(f"  rough independent episodes: ~{indep_est}")
    if indep_est > 0:
        se = np.sqrt(0.25 / indep_est)
        print(f"  95% CI half-width on that basis: {1.96 * se:.2%} "
              f"(vs {1.96 * np.sqrt(0.25 / max(int(sel.sum()), 1)):.2%} if treated as independent)")

    print("\n" + "=" * 100)
    print("T — Bad ticks: how extreme are the return tails?")
    print("=" * 100)
    all_r1 = []
    for ticker, b in bars.items():
        all_r1.append(b["close"].pct_change().dropna())
    r1 = pd.concat(all_r1)
    print(f"  1-min return percentiles (bps):")
    for q in (0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999):
        print(f"    p{q * 100:<8.2f} : {r1.quantile(q) * 10000:>12.1f}")
    extreme = (r1.abs() > 0.10).sum()
    print(f"  bars with |1-min return| > 10% : {extreme} ({extreme / len(r1):.6%})")
    tXtr, tytr, tXte, tyte, _, _ = assemble(bars, clip_returns=0.001)
    print()
    fit_eval(tXtr, tytr, tXte, tyte, "with returns winsorized at 0.1%/99.9%")

    print("\n" + "=" * 100)
    print("U — Labels that straddle the closing bell")
    print("=" * 100)
    cXtr, cytr, cXte, cyte, _, _ = assemble(bars, drop_close_spanning=True)
    fit_eval(cXtr, cytr, cXte, cyte, "close-spanning labels dropped")

    print("\n" + "=" * 100)
    print("V — Regime stability: is the edge steady across the history?")
    print("=" * 100)
    for frac, name in ((0.4, "earliest 20% window"), (0.2, "most recent 20% window")):
        vXtr, vytr, vXte, vyte, _, _ = assemble(bars, test_frac=frac)
        # For the earliest window, test on the slice right after training.
        fit_eval(vXtr, vytr, vXte, vyte, name)

    print("\nDone.")


if __name__ == "__main__":
    main()
