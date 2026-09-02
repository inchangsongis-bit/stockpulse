"""
Second research pass on the 5-minute forecast (see
research_forecast_improvements.py for the first, which found that neither
more OHLCV-derived features nor a different gradient-boosting library
helped). This pass tests structural hypotheses about the DATA rather than
the model:

  A. Regular-trading-hours only. ~35% of our minute bars are pre/post
     market, where per-bar volume is 10-100x thinner (AAPL: ~800-6k
     shares/min extended vs ~45-107k during RTH). Those bars are
     dominated by bid-ask bounce rather than price discovery, so they may
     be diluting whatever signal exists rather than adding to it.

  B. Market-context features. Every prediction is currently made for one
     ticker in complete isolation, but intraday single-stock returns are
     heavily driven by market beta — and the comovement literature finds
     that correlation rises through the session. The cross-sectional mean
     return across all watchlist tickers at each timestamp is a free
     market-factor proxy computable from data we already have. (No
     lookahead: the market's return over the last N minutes is known at
     prediction time t; we predict t+5.)

  C. Both together.

  D. Diagnostic — is the model usefully uncertain, or uniformly noisy?
     Accuracy stratified by prediction confidence. A model that's right
     52% of the time overall but 58% on its most confident decile is a
     usable product (act only on high-confidence calls); one that's flat
     across deciles is not.

  E. Diagnostic — predict market-RELATIVE (idiosyncratic) return instead
     of absolute. If absolute direction is unpredictable mostly because
     market-wide moves swamp it, the residual may be more learnable.
     Note this is a different product question, reported for insight
     only.

  F. Feature importance (permutation) on the best configuration.

Run: cd backend && source venv/bin/activate && python scripts/research_forecast_v2.py
"""

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.forecast_features import (  # noqa: E402
    FEATURE_COLUMNS as BASE_FEATURES,
    HORIZON_MINUTES,
    build_features,
    build_labels,
)
from database import async_session  # noqa: E402
from models import OHLCV  # noqa: E402

# US regular trading hours in the machine's local timezone (Pacific):
# 09:30-16:00 ET == 06:30-13:00 PT == minute-of-day 390..780.
RTH_START_MIN, RTH_END_MIN = 390, 780

MARKET_FEATURES = ["mkt_ret_1", "mkt_ret_5", "rel_strength_1", "rel_strength_5"]


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
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    return df[(minute_of_day >= RTH_START_MIN) & (minute_of_day <= RTH_END_MIN)].reset_index(drop=True)


def build_market_factor(bars_by_ticker: dict) -> pd.DataFrame:
    """
    Cross-sectional mean 1-min and 5-min return across all tickers, indexed
    by timestamp — a free market-factor proxy from data we already have.
    """
    per_ticker = []
    for ticker, bars in bars_by_ticker.items():
        d = bars[["timestamp"]].copy()
        d["r1"] = bars["close"].pct_change(1)
        d["r5"] = bars["close"].pct_change(5)
        per_ticker.append(d)
    stacked = pd.concat(per_ticker, ignore_index=True)
    mkt = stacked.groupby("timestamp")[["r1", "r5"]].mean().reset_index()
    return mkt.rename(columns={"r1": "mkt_ret_1", "r5": "mkt_ret_5"})


def assemble(bars_by_ticker: dict, rth_only: bool, with_market: bool, relative_target: bool = False):
    """Returns (X_train, y_train, X_test, y_test, feature_columns) using a
    per-ticker chronological 80/20 split."""
    if rth_only:
        bars_by_ticker = {t: filter_rth(b) for t, b in bars_by_ticker.items()}
        bars_by_ticker = {t: b for t, b in bars_by_ticker.items() if len(b) > 100}

    features = list(BASE_FEATURES)
    mkt = build_market_factor(bars_by_ticker) if (with_market or relative_target) else None
    if with_market:
        features += MARKET_FEATURES

    tr_X, tr_y, te_X, te_y = [], [], [], []
    for ticker, bars in bars_by_ticker.items():
        if len(bars) < 60:
            continue
        featured = build_features(bars)

        if mkt is not None:
            featured = featured.merge(mkt, on="timestamp", how="left")
            featured["rel_strength_1"] = featured["ret_1"] - featured["mkt_ret_1"]
            featured["rel_strength_5"] = featured["ret_5"] - featured["mkt_ret_5"]

        if relative_target:
            # Idiosyncratic label: did this ticker OUTperform the market
            # over the next 5 minutes (rather than simply rise)?
            close = bars["close"].reset_index(drop=True)
            ts = pd.to_datetime(bars["timestamp"]).reset_index(drop=True)
            fwd = (close.shift(-HORIZON_MINUTES) - close) / close
            gap = (ts.shift(-HORIZON_MINUTES) - ts).dt.total_seconds() / 60
            mkt_fwd = featured["mkt_ret_5"].shift(-HORIZON_MINUTES).reset_index(drop=True)
            labels = (fwd > mkt_fwd).astype(float)
            labels[(gap != HORIZON_MINUTES) | fwd.isna() | mkt_fwd.isna()] = np.nan
        else:
            labels = build_labels(bars)

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
        features,
    )


def run(label, X_train, y_train, X_test, y_test, return_model=False):
    model = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")
    print(f"{label:52s} train={len(X_train):>7d} test={len(X_test):>6d}  acc={acc:.4f}  AUC={auc:.4f}")
    if return_model:
        return model, proba, acc, auc
    return acc, auc


def confidence_stratified(proba, y_test):
    """Accuracy by decile of |proba - 0.5| — is the model usefully uncertain?"""
    conf = np.abs(proba - 0.5)
    preds = (proba >= 0.5).astype(int)
    correct = (preds == np.asarray(y_test)).astype(int)
    deciles = pd.qcut(conf, 10, labels=False, duplicates="drop")
    print(f"\n  {'confidence decile':<22s}{'n':>8s}{'accuracy':>11s}")
    for d in sorted(pd.unique(deciles)):
        mask = deciles == d
        print(f"  {'D' + str(int(d) + 1) + (' (least confident)' if d == 0 else ' (most confident)' if d == 9 else ''):<22s}"
              f"{mask.sum():>8d}{correct[mask].mean():>11.4f}")


def main():
    print("Loading minute bars...")
    bars_by_ticker = asyncio.run(load_minute_bars_by_ticker())
    print(f"  {len(bars_by_ticker)} tickers, {sum(len(b) for b in bars_by_ticker.values())} bars\n")

    print("=" * 96)
    print("A/B/C — data-structure hypotheses (all: per-ticker chronological split, same model)")
    print("=" * 96)

    Xtr, ytr, Xte, yte, _ = assemble(bars_by_ticker, rth_only=False, with_market=False)
    run("Baseline: all hours, no market context", Xtr, ytr, Xte, yte)

    Xtr, ytr, Xte, yte, _ = assemble(bars_by_ticker, rth_only=True, with_market=False)
    run("A. Regular trading hours only", Xtr, ytr, Xte, yte)

    Xtr, ytr, Xte, yte, _ = assemble(bars_by_ticker, rth_only=False, with_market=True)
    run("B. All hours + market-context features", Xtr, ytr, Xte, yte)

    Xtr, ytr, Xte, yte, feats = assemble(bars_by_ticker, rth_only=True, with_market=True)
    model, proba, acc, auc = run("C. RTH only + market context (combined)", Xtr, ytr, Xte, yte, return_model=True)

    print("\n" + "=" * 96)
    print("D — is the best model usefully uncertain? (accuracy by confidence decile)")
    print("=" * 96)
    confidence_stratified(proba, yte)

    print("\n" + "=" * 96)
    print("E — diagnostic: predict market-RELATIVE (idiosyncratic) direction instead")
    print("=" * 96)
    rXtr, rytr, rXte, ryte, _ = assemble(bars_by_ticker, rth_only=True, with_market=True, relative_target=True)
    run("E. RTH + market ctx, relative (vs-market) target", rXtr, rytr, rXte, ryte)

    print("\n" + "=" * 96)
    print("F — permutation feature importance on configuration C")
    print("=" * 96)
    sub = min(20000, len(Xte))
    imp = permutation_importance(
        model, Xte.iloc[:sub], yte.iloc[:sub], n_repeats=3, random_state=42, scoring="roc_auc"
    )
    order = np.argsort(imp.importances_mean)[::-1]
    print(f"  {'feature':<20s}{'AUC drop when shuffled':>26s}")
    for i in order:
        print(f"  {feats[i]:<20s}{imp.importances_mean[i]:>26.5f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
