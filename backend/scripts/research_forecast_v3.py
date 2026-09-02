"""
Third research pass. v1 tested model/feature variations, v2 tested data
STRUCTURE hypotheses (regular-hours filtering, market-context features) —
none improved on the baseline. This pass answers the two questions that
determine whether more effort on this is worth spending at all:

  G. LEARNING CURVE — does MORE data help? Train on 25/50/75/100% of the
     training rows and watch the test metric. If accuracy is still
     climbing at 100%, we're variance-limited and fetching more history
     (Polygon free tier can give ~2 years vs our current 29 days, at
     ~2.7hrs of rate-limited paginated fetching) would pay off. If it's
     flat, we're at a signal ceiling and more of the SAME data is wasted
     effort — we'd need a different KIND of data.

  H. HORIZON SWEEP — is 5 minutes just the hardest possible ask? Predict
     direction 5 / 15 / 30 / 60 / 120 minutes ahead with identical
     features and model. Short horizons are dominated by microstructure
     noise; if accuracy climbs with horizon, the product answer is to
     forecast a longer window rather than to keep grinding on 5 minutes.

  I. FEATURE PRUNING — v2's permutation importance found rsi_7 (-0.0042),
     volatility_10 and mkt_ret_1 have NEGATIVE importance (shuffling them
     IMPROVES AUC), meaning they're pure noise the model is overfitting
     to. Drop them and re-measure.

  J. EXTENDED-HOURS SANITY CHECK — v2 found RTH-only filtering made
     accuracy WORSE, which is suspicious: thin pre/post-market bars
     should carry less real information, not more. The likely explanation
     is bid-ask bounce, which creates statistically predictable negative
     autocorrelation that is NOT tradeable (you'd pay the spread every
     time). Measure 1-min return autocorrelation in extended vs regular
     hours to confirm — if extended hours is far more negatively
     autocorrelated, the baseline's edge is partly a microstructure
     artifact and the real tradeable accuracy is below the headline
     number.

Run: cd backend && source venv/bin/activate && python scripts/research_forecast_v3.py
"""

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.forecast_features import (  # noqa: E402
    FEATURE_COLUMNS as BASE_FEATURES,
    build_features,
)
from database import async_session  # noqa: E402
from models import OHLCV  # noqa: E402

RTH_START_MIN, RTH_END_MIN = 390, 780
PRUNED_FEATURES = [f for f in BASE_FEATURES if f not in ("rsi_7", "volatility_10")]


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


def labels_for_horizon(df: pd.DataFrame, horizon: int) -> pd.Series:
    close = df["close"]
    ts = pd.to_datetime(df["timestamp"])
    fwd = close.shift(-horizon)
    gap = (ts.shift(-horizon) - ts).dt.total_seconds() / 60
    label = (fwd > close).astype(float)
    label[gap != horizon] = np.nan
    return label


def assemble(bars_by_ticker: dict, horizon: int, features: list):
    tr_X, tr_y, te_X, te_y = [], [], [], []
    for ticker, bars in bars_by_ticker.items():
        if len(bars) < 60:
            continue
        featured = build_features(bars)
        labels = labels_for_horizon(bars, horizon)
        valid = featured[features].notna().all(axis=1) & labels.notna()
        X = featured.loc[valid, features].reset_index(drop=True)
        y = labels.loc[valid].reset_index(drop=True)
        if len(X) < 20:
            continue
        cut = int(len(X) * 0.8)
        tr_X.append(X.iloc[:cut]); tr_y.append(y.iloc[:cut])
        te_X.append(X.iloc[cut:]); te_y.append(y.iloc[cut:])
    return (
        pd.concat(tr_X, ignore_index=True), pd.concat(tr_y, ignore_index=True),
        pd.concat(te_X, ignore_index=True), pd.concat(te_y, ignore_index=True),
    )


def run(label, X_train, y_train, X_test, y_test):
    model = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")
    print(f"{label:48s} train={len(X_train):>7d} test={len(X_test):>6d}  acc={acc:.4f}  AUC={auc:.4f}")
    return acc, auc


def main():
    print("Loading minute bars...")
    bars_by_ticker = asyncio.run(load_minute_bars_by_ticker())
    print(f"  {len(bars_by_ticker)} tickers, {sum(len(b) for b in bars_by_ticker.values())} bars\n")

    print("=" * 92)
    print("G — LEARNING CURVE: does more of the SAME data help? (5-min horizon, base features)")
    print("=" * 92)
    Xtr, ytr, Xte, yte = assemble(bars_by_ticker, horizon=5, features=BASE_FEATURES)
    for frac in (0.25, 0.50, 0.75, 1.00):
        n = int(len(Xtr) * frac)
        # Take the most RECENT slice at each size — that's what having
        # more history would actually give us (more past, same present).
        run(f"  {int(frac * 100):>3d}% of training rows", Xtr.iloc[-n:], ytr.iloc[-n:], Xte, yte)

    print("\n" + "=" * 92)
    print("H — HORIZON SWEEP: is 5 minutes simply the hardest ask?")
    print("=" * 92)
    for horizon in (5, 15, 30, 60, 120):
        hXtr, hytr, hXte, hyte = assemble(bars_by_ticker, horizon=horizon, features=BASE_FEATURES)
        run(f"  {horizon:>3d}-minute-ahead direction", hXtr, hytr, hXte, hyte)

    print("\n" + "=" * 92)
    print("I — FEATURE PRUNING: drop the negative-importance features (rsi_7, volatility_10)")
    print("=" * 92)
    run("  Base feature set (7 features)", Xtr, ytr, Xte, yte)
    pXtr, pytr, pXte, pyte = assemble(bars_by_ticker, horizon=5, features=PRUNED_FEATURES)
    run(f"  Pruned feature set ({len(PRUNED_FEATURES)} features)", pXtr, pytr, pXte, pyte)

    print("\n" + "=" * 92)
    print("J — MICROSTRUCTURE CHECK: 1-min return autocorrelation, extended vs regular hours")
    print("=" * 92)
    print("   (Strong NEGATIVE autocorrelation = bid-ask bounce = statistically")
    print("    'predictable' but NOT tradeable, since you'd pay the spread each time.)")
    ext_acf, rth_acf = [], []
    for ticker, bars in bars_by_ticker.items():
        ts = pd.to_datetime(bars["timestamp"])
        mod = ts.dt.hour * 60 + ts.dt.minute
        is_rth = (mod >= RTH_START_MIN) & (mod <= RTH_END_MIN)
        r = bars["close"].pct_change()
        for mask, sink in ((is_rth, rth_acf), (~is_rth, ext_acf)):
            sub = r[mask].dropna()
            if len(sub) > 200:
                sink.append(sub.autocorr(lag=1))
    print(f"\n   Regular hours  : mean lag-1 autocorrelation = {np.mean(rth_acf):+.4f}  (n={len(rth_acf)} tickers)")
    print(f"   Extended hours : mean lag-1 autocorrelation = {np.mean(ext_acf):+.4f}  (n={len(ext_acf)} tickers)")

    print("\nDone.")


if __name__ == "__main__":
    main()
