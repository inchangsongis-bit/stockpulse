"""
Train the short-term (5-minute-ahead) direction classifier used by
analysis/forecast.py. Run manually whenever enough minute-bar history has
accumulated to be worth (re)training on:

    cd backend && source venv/bin/activate && python scripts/train_forecast_model.py

Pulls every ticker's minute bars from the local DB, builds features/labels
per analysis/forecast_features.py, evaluates on a proper per-ticker
chronological holdout (build_chronological_split — each ticker's own last
20% by time, not a single cut across the ticker-then-time-concatenated
array, which would mostly separate tickers rather than time), then
refits the same model on the FULL dataset (train + test) before saving,
so the shipped model isn't leaving each ticker's most recent ~20% of
data unused.

Model choice: HistGradientBoostingClassifier, chosen after an explicit
side-by-side against LightGBM on this exact data (scripts/
research_forecast_improvements.py) showed no meaningful difference —
consistent with general findings that gradient-boosting library choice
barely moves the needle on weak-signal tabular problems like this one.
An expanded feature set (more technical/volume indicators) was tried the
same way and didn't help either — see that script's docstring and the
project's notes on this for the full comparison and why: a 5-minute-
ahead direction call on 1-minute equity bars is close to a coin flip by
nature (consistent with the efficient-market/random-walk literature),
and the strongest real signal at this horizon — order flow imbalance
from level-2 order book data — isn't available from Polygon for stocks
at any tier, only for crypto. Meaningfully beating ~52% here would need
a different (paid) data source, not more feature engineering on OHLCV.

This is a probability lean, not a reliable prediction, and the app's
UI/email output should always present it that way.
"""

import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.forecast import MODEL_DIR, MODEL_PATH  # noqa: E402
from analysis.forecast_features import FEATURE_COLUMNS, build_chronological_split  # noqa: E402
from database import async_session  # noqa: E402
from models import OHLCV  # noqa: E402


async def load_minute_bars_by_ticker() -> dict:
    async with async_session() as db:
        result = await db.execute(
            select(OHLCV)
            .where(OHLCV.interval == "minute")
            .order_by(OHLCV.ticker, OHLCV.timestamp.asc())
        )
        rows = result.scalars().all()

    bars_by_ticker = defaultdict(list)
    for r in rows:
        bars_by_ticker[r.ticker].append({
            "timestamp": r.timestamp,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
        })

    return {t: pd.DataFrame(bars) for t, bars in bars_by_ticker.items()}


def main():
    print("Loading minute bars from the database...")
    bars_by_ticker = asyncio.run(load_minute_bars_by_ticker())
    print(f"  {len(bars_by_ticker)} tickers with minute data")

    print("Building features + labels with a per-ticker chronological split...")
    X_train, y_train, X_test, y_test = build_chronological_split(bars_by_ticker, test_frac=0.2)
    n_total = len(X_train) + len(X_test)
    print(f"  {n_total} labeled rows after warmup/session-gap filtering "
          f"({len(X_train)} train, {len(X_test)} test)")

    if n_total < 500:
        print(
            "Not enough labeled rows to train a meaningful model "
            "(need at least 500). Sync more minute data first."
        )
        return

    print(f"Evaluating on the held-out chronological test set ({len(X_test)} rows)...")
    eval_model = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    eval_model.fit(X_train, y_train)
    proba = eval_model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")  # only one class present in y_test

    print(f"  Test accuracy: {accuracy:.3f}  (baseline coin-flip: 0.500)")
    print(f"  Test AUC:      {auc:.3f}  (baseline coin-flip: 0.500)")

    # Conviction thresholds. Research (scripts/research_forecast_v5.py,
    # v6.py) found accuracy is NOT uniform across predictions: it rises
    # monotonically with the model's own confidence, and — importantly —
    # so does the size of the move being predicted (avg |5-min move| of
    # 11bps across all predictions vs 31bps in the top 1%). Those
    # percentile cutoffs are recorded here so the API can tell a
    # genuinely high-conviction call apart from a coin-flip one, rather
    # than presenting every prediction as equally meaningful.
    conf = np.abs(proba - 0.5)
    high_conviction_cut = float(np.percentile(conf, 99))
    moderate_conviction_cut = float(np.percentile(conf, 90))

    high_mask = conf >= high_conviction_cut
    high_acc = accuracy_score(y_test[high_mask], preds[high_mask]) if high_mask.sum() else float("nan")
    print(f"  Top 1% most-confident accuracy:  {high_acc:.3f}  (n={int(high_mask.sum())})")

    # Refit on everything (train + test) for the model actually shipped —
    # the held-out fold above is only to get an honest accuracy estimate,
    # not to withhold each ticker's most recent ~20% from the live model.
    print(f"Refitting on the full {n_total}-row dataset for the model to ship...")
    model = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)
    model.fit(X_full, y_full)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "n_tickers": len(bars_by_ticker),
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "test_accuracy": round(float(accuracy), 4),
        "test_auc": round(float(auc), 4) if auc == auc else None,
        "test_accuracy_high_conviction": round(float(high_acc), 4) if high_acc == high_acc else None,
        "high_conviction_cut": round(high_conviction_cut, 6),
        "moderate_conviction_cut": round(moderate_conviction_cut, 6),
        "feature_columns": FEATURE_COLUMNS,
        "split_method": "per_ticker_chronological",
    }
    metadata_path = MODEL_PATH.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
