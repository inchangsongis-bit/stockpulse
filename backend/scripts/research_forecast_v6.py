"""
Robustness check on the one finding that mattered.

v5 found that the model's edge is unusable across all predictions
(50.8% win rate, +0.46bps gross, loses money at any realistic cost) but
becomes economically viable when restricted to its most confident calls:

    top 10%  -> 52.78% win rate, avg move 17.96bps, +1.86bps gross
    top  1%  -> 56.30% win rate, avg move 30.59bps, +6.55bps gross

The mechanism is that confidence correlates with move SIZE as well as
direction — the model finds moments that are both volatile and
directional, and it's the product of the two that pays.

That's a big claim resting on 778 trades from one ~6-day held-out window,
selected post-hoc as the best of several configurations. Before acting on
it, this script checks whether it survives:

  N. Statistical significance — binomial test of the top-1% win rate
     against a 50/50 null.
  O. Time stability — split the held-out window in half chronologically
     and check both halves independently. A real effect should appear in
     both; a fluke usually lives in one.
  P. Seed stability — retrain with several random seeds. If the top-1%
     selection is driven by model variance rather than signal, the
     result will swing wildly across seeds.
  Q. Ticker concentration — is the edge spread across the watchlist, or
     is it one or two tickers carrying the whole result?

Run: cd backend && source venv/bin/activate && python scripts/research_forecast_v6.py
"""

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.forecast_features import FEATURE_COLUMNS as BASE_FEATURES, build_features  # noqa: E402
from database import async_session  # noqa: E402
from models import OHLCV  # noqa: E402

RTH_START_MIN, RTH_END_MIN = 390, 780
HORIZON = 5


async def load_minute_bars_by_ticker() -> dict:
    async with async_session() as db:
        result = await db.execute(
            select(OHLCV).where(OHLCV.interval == "minute").order_by(OHLCV.ticker, OHLCV.timestamp.asc())
        )
        rows = result.scalars().all()
    bars = defaultdict(list)
    for r in rows:
        bars[r.ticker].append({
            "timestamp": r.timestamp, "open": r.open, "high": r.high,
            "low": r.low, "close": r.close, "volume": r.volume,
        })
    return {t: pd.DataFrame(b) for t, b in bars.items()}


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    mod = ts.dt.hour * 60 + ts.dt.minute
    return df[(mod >= RTH_START_MIN) & (mod <= RTH_END_MIN)].reset_index(drop=True)


def build_dataset(bars_by_ticker):
    tr_X, tr_y = [], []
    te_X, te_y, te_ret, te_tick, te_ts = [], [], [], [], []
    for ticker, bars in bars_by_ticker.items():
        featured = build_features(bars)
        close = bars["close"]
        ts = pd.to_datetime(bars["timestamp"])
        fwd_ret = (close.shift(-HORIZON) - close) / close
        gap = (ts.shift(-HORIZON) - ts).dt.total_seconds() / 60
        fwd_ret[gap != HORIZON] = np.nan
        labels = (fwd_ret > 0).astype(float)
        labels[fwd_ret.isna()] = np.nan

        valid = featured[BASE_FEATURES].notna().all(axis=1) & labels.notna()
        X = featured.loc[valid, BASE_FEATURES].reset_index(drop=True)
        y = labels.loc[valid].reset_index(drop=True)
        r = fwd_ret.loc[valid].reset_index(drop=True)
        t = ts.loc[valid].reset_index(drop=True)
        if len(X) < 30:
            continue
        cut = int(len(X) * 0.8)
        tr_X.append(X.iloc[:cut]); tr_y.append(y.iloc[:cut])
        te_X.append(X.iloc[cut:]); te_y.append(y.iloc[cut:])
        te_ret.append(r.iloc[cut:]); te_ts.append(t.iloc[cut:])
        te_tick.append(pd.Series([ticker] * (len(X) - cut)))

    return (
        pd.concat(tr_X, ignore_index=True), pd.concat(tr_y, ignore_index=True),
        pd.concat(te_X, ignore_index=True), pd.concat(te_y, ignore_index=True),
        pd.concat(te_ret, ignore_index=True).to_numpy() * 10000,
        pd.concat(te_tick, ignore_index=True).to_numpy(),
        pd.concat(te_ts, ignore_index=True).to_numpy(),
    )


def top_pct_stats(proba, ret, pct=99):
    conf = np.abs(proba - 0.5)
    mask = conf >= np.percentile(conf, pct)
    direction = np.where(proba[mask] >= 0.5, 1.0, -1.0)
    r = ret[mask]
    win = (np.sign(direction) == np.sign(r)).mean()
    gross = (direction * r).mean()
    return mask, win, gross, mask.sum()


def main():
    print("Loading regular-hours minute bars...")
    bars = asyncio.run(load_minute_bars_by_ticker())
    bars = {t: filter_rth(b) for t, b in bars.items()}
    bars = {t: b for t, b in bars.items() if len(b) > 200}
    X_train, y_train, X_test, y_test, ret_test, tick_test, ts_test = build_dataset(bars)
    print(f"  {len(bars)} tickers | train={len(X_train)} test={len(X_test)}\n")

    model = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    print("=" * 92)
    print("N — Statistical significance of the top-1% win rate")
    print("=" * 92)
    mask, win, gross, n = top_pct_stats(proba, ret_test, 99)
    wins = int(round(win * n))
    pval = stats.binomtest(wins, n, 0.5, alternative="greater").pvalue
    se = np.sqrt(0.25 / n)
    lo, hi = win - 1.96 * se, win + 1.96 * se
    print(f"  trades          : {n}")
    print(f"  win rate        : {win:.2%}   (95% CI {lo:.2%} - {hi:.2%})")
    print(f"  one-sided p-val : {pval:.5f}  vs a 50/50 null")
    print(f"  verdict         : {'statistically significant' if pval < 0.05 else 'NOT significant'}")
    print("  caveat          : selected post-hoc as the best of several configurations,")
    print("                    so the effective p-value is weaker than it looks.")

    print("\n" + "=" * 92)
    print("O — Time stability: does it hold in BOTH halves of the held-out window?")
    print("=" * 92)
    median_ts = np.median(ts_test.astype("datetime64[ns]").astype(np.int64))
    first_half = ts_test.astype("datetime64[ns]").astype(np.int64) < median_ts
    for name, half in (("first half ", first_half), ("second half", ~first_half)):
        p, r = proba[half], ret_test[half]
        conf = np.abs(p - 0.5)
        m = conf >= np.percentile(conf, 99)
        d = np.where(p[m] >= 0.5, 1.0, -1.0)
        w = (np.sign(d) == np.sign(r[m])).mean()
        g = (d * r[m]).mean()
        print(f"  {name} : n={m.sum():>5d}  win rate={w:>7.2%}  gross={g:>+7.3f} bps")

    print("\n" + "=" * 92)
    print("P — Seed stability: top-1% across 5 random seeds")
    print("=" * 92)
    wins_across_seeds = []
    for seed in (0, 1, 42, 123, 2024):
        m = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=seed)
        m.fit(X_train, y_train)
        pr = m.predict_proba(X_test)[:, 1]
        _, w, g, n_ = top_pct_stats(pr, ret_test, 99)
        wins_across_seeds.append(w)
        print(f"  seed {seed:>5d} : n={n_:>5d}  win rate={w:>7.2%}  gross={g:>+7.3f} bps")
    print(f"\n  across seeds: mean={np.mean(wins_across_seeds):.2%}  "
          f"std={np.std(wins_across_seeds):.2%}  "
          f"min={min(wins_across_seeds):.2%}  max={max(wins_across_seeds):.2%}")

    print("\n" + "=" * 92)
    print("Q — Ticker concentration: is one name carrying the whole result?")
    print("=" * 92)
    mask, _, _, _ = top_pct_stats(proba, ret_test, 99)
    sel_ticks = tick_test[mask]
    d = np.where(proba[mask] >= 0.5, 1.0, -1.0)
    pnl = d * ret_test[mask]
    df = pd.DataFrame({"ticker": sel_ticks, "pnl": pnl})
    by_ticker = df.groupby("ticker")["pnl"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False)
    print(f"  {len(by_ticker)} distinct tickers in the top-1% selection")
    print(f"\n  Top 5 contributors by total P&L:")
    print(by_ticker.head(5).to_string())
    total = by_ticker["sum"].sum()
    top1_share = by_ticker["sum"].iloc[0] / total if total else float("nan")
    print(f"\n  Single largest ticker's share of total P&L: {top1_share:.1%}")
    print(f"  (A healthy, generalizable edge is spread across many names;")
    print(f"   one ticker dominating means it's really a single-stock artifact.)")

    print("\nDone.")


if __name__ == "__main__":
    main()
