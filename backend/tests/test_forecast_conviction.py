import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from analysis import forecast


def make_bars(n=30, start=100.0):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-02 09:30", periods=n, freq="min"),
        "open": [start + i * 0.01 for i in range(n)],
        "high": [start + i * 0.01 + 0.02 for i in range(n)],
        "low": [start + i * 0.01 - 0.02 for i in range(n)],
        "close": [start + i * 0.01 for i in range(n)],
        "volume": [100_000] * n,
    })


@pytest.fixture(autouse=True)
def _clear_caches():
    forecast._load_model.cache_clear()
    forecast._load_conviction_cuts.cache_clear()
    yield
    forecast._load_model.cache_clear()
    forecast._load_conviction_cuts.cache_clear()


def _mock_model(monkeypatch, proba_up):
    model = MagicMock()
    model.predict_proba.return_value = [[1 - proba_up, proba_up]]
    monkeypatch.setattr(forecast, "_load_model", lambda: model)


def _mock_cuts(monkeypatch, tmp_path, moderate, high):
    meta = tmp_path / "forecast_model.json"
    meta.write_text(json.dumps({"moderate_conviction_cut": moderate, "high_conviction_cut": high}))
    monkeypatch.setattr(forecast, "METADATA_PATH", meta)


@pytest.mark.parametrize(
    "proba_up,expected",
    [
        (0.50, "low"),        # edge 0.000 — a pure coin flip
        (0.52, "low"),        # edge 0.020 — below the moderate cut
        (0.54, "moderate"),   # edge 0.040 — between the cuts
        (0.60, "high"),       # edge 0.100 — above the high cut
        (0.40, "high"),       # edge 0.100 the other direction
    ],
)
def test_conviction_tiers_from_recorded_cuts(monkeypatch, tmp_path, proba_up, expected):
    _mock_model(monkeypatch, proba_up)
    _mock_cuts(monkeypatch, tmp_path, moderate=0.028, high=0.052)

    result = forecast.predict_direction(make_bars(), sentiment=0.0)

    assert result["conviction"] == expected


def test_conviction_defaults_to_low_when_metadata_missing(monkeypatch, tmp_path):
    _mock_model(monkeypatch, 0.99)  # would be "high" if cuts were known
    monkeypatch.setattr(forecast, "METADATA_PATH", tmp_path / "does_not_exist.json")

    result = forecast.predict_direction(make_bars(), sentiment=0.0)

    # Without recorded cuts there's no basis to claim conviction, so it
    # must not overstate — even for an extreme probability.
    assert result["conviction"] == "low"


def test_conviction_survives_corrupt_metadata(monkeypatch, tmp_path):
    _mock_model(monkeypatch, 0.60)
    meta = tmp_path / "forecast_model.json"
    meta.write_text("{not valid json")
    monkeypatch.setattr(forecast, "METADATA_PATH", meta)

    result = forecast.predict_direction(make_bars(), sentiment=0.0)
    assert result["conviction"] == "low"


def test_sentiment_nudge_can_change_conviction_tier(monkeypatch, tmp_path):
    # Model alone says 0.54 (edge 0.04 -> moderate); a strongly positive
    # sentiment nudge (+1.0 * 0.05) pushes it to 0.59 (edge 0.09 -> high).
    _mock_model(monkeypatch, 0.54)
    _mock_cuts(monkeypatch, tmp_path, moderate=0.028, high=0.052)

    plain = forecast.predict_direction(make_bars(), sentiment=0.0)
    nudged = forecast.predict_direction(make_bars(), sentiment=1.0)

    assert plain["conviction"] == "moderate"
    assert nudged["conviction"] == "high"
    assert nudged["model_probability_up"] == 0.54  # raw model output unchanged
