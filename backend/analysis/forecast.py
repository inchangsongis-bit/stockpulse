"""
Loads the trained short-term direction model (see
scripts/train_forecast_model.py) and predicts P(price up) for a ticker's
next 5 minutes from its most recent minute bars, then blends in a small,
bounded adjustment from the ticker's current news sentiment — the model
itself was never trained on sentiment (see forecast_features.py's
docstring for why: there's no historical per-minute sentiment series to
train against, only ever a current snapshot).
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from analysis.forecast_features import FEATURE_COLUMNS, build_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "ml"
MODEL_PATH = MODEL_DIR / "forecast_model.joblib"
METADATA_PATH = MODEL_PATH.with_suffix(".json")

# How much the current sentiment score (-1..+1) can nudge the model's own
# probability — deliberately small; this is a tiebreaker, not a second
# vote.
SENTIMENT_WEIGHT = 0.05

# US regular trading hours as minute-of-day in the machine's local
# timezone — the same frame the stored bar timestamps use, since
# data_sources/polygon.py converts with datetime.fromtimestamp().
RTH_START_MIN, RTH_END_MIN = 390, 780
EDGE_WINDOW_MINUTES = 30

# Error analysis over 2 years / 13.2M bars (scripts/research_forecast_v8.py)
# found the model is at its worst in the opening and closing half-hours:
# 50.6% accuracy there versus 51.8% through the middle of the session,
# while those windows carry the LARGEST moves (26.5bps average vs 14.3)
# and 22.5% of all high-conviction calls. That combination — biggest
# exposure exactly where the model is least skilled — is worth declining
# rather than acting on, so conviction is capped during those windows.


class ForecastUnavailable(Exception):
    pass


def _in_session_edge_window(ts) -> bool:
    minute_of_day = ts.hour * 60 + ts.minute
    return (
        RTH_START_MIN <= minute_of_day < RTH_START_MIN + EDGE_WINDOW_MINUTES
        or RTH_END_MIN - EDGE_WINDOW_MINUTES < minute_of_day <= RTH_END_MIN
    )


@lru_cache
def _load_model():
    if not MODEL_PATH.exists():
        raise ForecastUnavailable(
            "No trained forecast model found — run "
            "`python scripts/train_forecast_model.py` first."
        )
    return joblib.load(MODEL_PATH)


@lru_cache
def _load_conviction_cuts() -> tuple:
    """
    (moderate_cut, high_cut) on |P(up) - 0.5|, recorded at training time.

    Accuracy is far from uniform across predictions: on the held-out set
    it runs ~50% in the least-confident decile and ~53-56% in the top
    decile/percentile, and the SIZE of the predicted move rises with
    confidence too (~11bps average across all predictions vs ~31bps in
    the top 1%). Most predictions this model makes are therefore
    coin-flips that shouldn't be presented as calls at all — these cuts
    let the API say so.
    """
    if not METADATA_PATH.exists():
        return (float("inf"), float("inf"))  # unknown -> label everything "low"
    try:
        meta = json.loads(METADATA_PATH.read_text())
        return (
            float(meta.get("moderate_conviction_cut", float("inf"))),
            float(meta.get("high_conviction_cut", float("inf"))),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return (float("inf"), float("inf"))


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
    edge = abs(adjusted - 0.5)
    confidence = round(edge * 200, 1)

    moderate_cut, high_cut = _load_conviction_cuts()
    if edge >= high_cut:
        conviction = "high"
    elif edge >= moderate_cut:
        conviction = "moderate"
    else:
        conviction = "low"

    # See _in_session_edge_window: the model is measurably least accurate
    # in the first and last half-hour, on the largest moves. Cap rather
    # than zero it out, so the call is still shown — just never dressed up
    # as high conviction.
    session_edge = _in_session_edge_window(featured.iloc[-1]["timestamp"])
    if session_edge and conviction == "high":
        conviction = "moderate"

    return {
        "direction": direction,
        "probability_up": round(adjusted, 3),
        "model_probability_up": round(proba_up, 3),
        "confidence": confidence,
        "conviction": conviction,
        "session_edge_window": session_edge,
        "horizon_minutes": 5,
    }
