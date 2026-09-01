"""
Loads the trained short-term direction model (see
scripts/train_forecast_model.py) and predicts P(price up) for a ticker's
next 5 minutes from its most recent minute bars, then blends in a small,
bounded adjustment from the ticker's current news sentiment — the model
itself was never trained on sentiment (see forecast_features.py's
docstring for why: there's no historical per-minute sentiment series to
train against, only ever a current snapshot).
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from analysis.forecast_features import FEATURE_COLUMNS, build_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "ml"
MODEL_PATH = MODEL_DIR / "forecast_model.joblib"

# How much the current sentiment score (-1..+1) can nudge the model's own
# probability — deliberately small; this is a tiebreaker, not a second
# vote.
SENTIMENT_WEIGHT = 0.05


class ForecastUnavailable(Exception):
    pass


@lru_cache
def _load_model():
    if not MODEL_PATH.exists():
        raise ForecastUnavailable(
            "No trained forecast model found — run "
            "`python scripts/train_forecast_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_direction(minute_bars: pd.DataFrame, sentiment: Optional[float] = 0.0) -> dict:
    """
    minute_bars: recent minute OHLCV for one ticker, ascending by
    timestamp — needs enough rows for the longest rolling feature window
    to fill up (20 bars) plus at least one more for the actual prediction
    row.
    sentiment: the ticker's current composite sentiment score, [-1, 1].
    """
    model = _load_model()

    featured = build_features(minute_bars).dropna(subset=FEATURE_COLUMNS)
    if featured.empty:
        raise ForecastUnavailable("Not enough recent minute bars to compute features.")

    latest = featured.iloc[[-1]][FEATURE_COLUMNS]
    proba_up = float(model.predict_proba(latest)[0][1])

    nudge = (sentiment or 0.0) * SENTIMENT_WEIGHT
    adjusted = max(0.0, min(1.0, proba_up + nudge))

    direction = "up" if adjusted >= 0.5 else "down"
    # 0 at a coin-flip (50/50), 100 at full certainty either direction.
    confidence = round(abs(adjusted - 0.5) * 200, 1)

    return {
        "direction": direction,
        "probability_up": round(adjusted, 3),
        "model_probability_up": round(proba_up, 3),
        "confidence": confidence,
        "horizon_minutes": 5,
    }
