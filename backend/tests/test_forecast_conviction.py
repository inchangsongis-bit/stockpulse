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


def _bars_ending_at(hour, minute, n=30):
    end = pd.Timestamp(2026, 1, 2, hour, minute)
    idx = pd.date_range(end=end, periods=n, freq="min")
    return pd.DataFrame({
        "timestamp": idx,
        "open": [100 + i * 0.01 for i in range(n)],
        "high": [100 + i * 0.01 + 0.02 for i in range(n)],
        "low": [100 + i * 0.01 - 0.02 for i in range(n)],
        "close": [100 + i * 0.01 for i in range(n)],
        "volume": [100_000] * n,
    })


@pytest.mark.parametrize(
    "hour,minute,expected_edge",
    [
        (6, 45, True),    # first half-hour after the 06:30 open
        (12, 50, True),   # last half-hour before the 13:00 close
        (10, 0, False),   # mid-session
        (8, 15, False),   # mid-session
    ],
)
def test_session_edge_window_detection(monkeypatch, tmp_path, hour, minute, expected_edge):
    _mock_model(monkeypatch, 0.60)  # strong enough to be "high" on its own
    _mock_cuts(monkeypatch, tmp_path, moderate=0.028, high=0.052)

    result = forecast.predict_direction(_bars_ending_at(hour, minute), sentiment=0.0)

    assert result["session_edge_window"] is expected_edge
    # High conviction is capped to moderate inside the open/close windows,
    # where the model measurably underperforms on the largest moves.
    assert result["conviction"] == ("moderate" if expected_edge else "high")


def test_session_edge_window_does_not_upgrade_low_conviction(monkeypatch, tmp_path):
    _mock_model(monkeypatch, 0.505)  # edge 0.005 -> "low"
    _mock_cuts(monkeypatch, tmp_path, moderate=0.028, high=0.052)

    result = forecast.predict_direction(_bars_ending_at(6, 45), sentiment=0.0)

    assert result["session_edge_window"] is True
    assert result["conviction"] == "low"  # capping must never raise a tier


@pytest.mark.parametrize(
    "hour,minute,expected_session,expected_conviction",
    [
        (10, 0, "regular", "high"),       # mid-session: full tiering
        (6, 45, "regular", "moderate"),   # opening half-hour: capped
        (12, 50, "regular", "moderate"),  # closing half-hour: capped
        (15, 0, "extended", "low"),       # after the close: floored
        (5, 0, "extended", "low"),        # pre-market: floored
    ],
)
def test_conviction_is_floored_outside_regular_hours(
    monkeypatch, tmp_path, hour, minute, expected_session, expected_conviction
):
    _mock_model(monkeypatch, 0.60)  # would be "high" on the raw score alone
    _mock_cuts(monkeypatch, tmp_path, moderate=0.028, high=0.052)

    result = forecast.predict_direction(_bars_ending_at(hour, minute), sentiment=0.0)

    assert result["market_session"] == expected_session
    # Extended-hours accuracy is bid-ask-bounce artifact (61.6% all-hours
    # vs 52.9% regular-hours at the top 0.1% of confidence), so conviction
    # must never advertise an edge there.
    assert result["conviction"] == expected_conviction
