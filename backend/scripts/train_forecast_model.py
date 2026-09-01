"""
Train the short-term (5-minute-ahead) direction classifier used by
analysis/forecast.py. Run manually whenever enough minute-bar history has
accumulated to be worth (re)training on:

    cd backend && source venv/bin/activate && python scripts/train_forecast_model.py

Pulls every ticker's minute bars from the local DB, builds features/labels
per analysis/forecast_features.py, pools them into one dataset (see that
module's docstring for why one pooled model rather than per-ticker), does
a time-ordered train/test split (not random — random would leak
adjacent-row rolling-window information across the split), trains a
HistGradientBoostingClassifier, and reports honest evaluation metrics
before saving the model.

A 5-minutes-ahead direction call on 1-minute equity bars is close to a
coin flip by nature — don't expect anything like a headline classifier's
accuracy. This is a probability lean, not a reliable prediction, and the
app's UI/email output should always present it that way.
"""

import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.forecast import MODEL_DIR, MODEL_PATH  # noqa: E402
from analysis.forecast_features import FEATURE_COLUMNS, build_training_set  # noqa: E402
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

    print("Building features + labels...")
    X, y = build_training_set(bars_by_ticker)
    print(f"  {len(X)} labeled rows after warmup/session-gap filtering")

    if len(X) < 500:
        print(
            "Not enough labeled rows to train a meaningful model "
            "(need at least 500). Sync more minute data first."
        )
        return

    # Time-ordered split (not shuffled/random): random splitting would let
    # rolling-window features from test-set rows leak information about
    # adjacent training rows. A simple 80/20 split on the pooled,
    # ticker-then-time-ordered data is an approximation, not a strict
    # walk-forward validation, but avoids the worst of that leakage.
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"Training HistGradientBoostingClassifier on {len(X_train)} rows...")
    model = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")  # only one class present in y_test

    print(f"  Test accuracy: {accuracy:.3f}  (baseline coin-flip: 0.500)")
    print(f"  Test AUC:      {auc:.3f}  (baseline coin-flip: 0.500)")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "n_tickers": len(bars_by_ticker),
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "test_accuracy": round(float(accuracy), 4),
        "test_auc": round(float(auc), 4) if auc == auc else None,
        "feature_columns": FEATURE_COLUMNS,
    }
    metadata_path = MODEL_PATH.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
