import numpy as np
import pandas as pd
import pytest

from analysis.options_features import (
    OPTIONS_FEATURE_COLUMNS,
    _bs_price,
    build_options_features,
    implied_vol,
)


def test_implied_vol_recovers_the_volatility_used_to_price_the_option():
    # Round trip: price an option at a known sigma, then invert it back.
    spot, strike, tau, sigma = 500.0, 500.0, 0.05, 0.22
    price = _bs_price(spot, strike, tau, 0.04, sigma, "call")
    recovered = implied_vol(price, spot, strike, tau, "call")
    assert recovered == pytest.approx(sigma, abs=1e-3)


def test_implied_vol_round_trips_for_puts_too():
    spot, strike, tau, sigma = 500.0, 495.0, 0.08, 0.31
    price = _bs_price(spot, strike, tau, 0.04, sigma, "put")
    assert implied_vol(price, spot, strike, tau, "put") == pytest.approx(sigma, abs=1e-3)


@pytest.mark.parametrize(
    "price,spot,strike,tau,opt_type",
    [
        (0.01, 500.0, 500.0, 0.05, "call"),   # below the noise floor
        (np.nan, 500.0, 500.0, 0.05, "call"),  # missing quote
        (10.0, 500.0, 500.0, -0.01, "call"),   # already expired
        (2.0, 500.0, 480.0, 0.05, "call"),     # price below intrinsic (20.0)
    ],
)
def test_implied_vol_returns_nan_rather_than_guessing(price, spot, strike, tau, opt_type):
    assert np.isnan(implied_vol(price, spot, strike, tau, opt_type))


def _chain(timestamps, spot=500.0):
    """A small two-strike call/put chain over the given minutes."""
    rows = []
    for ts in timestamps:
        for opt_type, strike, close, volume in [
            ("call", 500.0, 6.0, 100),
            ("put", 500.0, 5.0, 250),
            ("call", 520.0, 1.0, 10),
        ]:
            rows.append({
                "contract": f"O:SPY{opt_type}{int(strike)}",
                "expiration": "2026-08-21", "strike": strike, "opt_type": opt_type,
                "timestamp": ts, "close": close, "volume": volume,
            })
    return pd.DataFrame(rows)


def test_build_options_features_computes_put_call_ratio():
    ts = pd.date_range("2026-08-14 10:00", periods=3, freq="min")
    opts = _chain(ts)
    spot = pd.DataFrame({"timestamp": ts, "close": 500.0})

    out = build_options_features(opts, spot)

    assert not out.empty
    for col in OPTIONS_FEATURE_COLUMNS:
        assert col in out.columns
    # 250 put contracts against 110 call contracts (100 at 500 + 10 at 520)
    assert out["put_call_volume_ratio"].iloc[0] == pytest.approx(250 / 110)


def test_build_options_features_returns_empty_when_no_overlapping_minutes():
    opts = _chain(pd.date_range("2026-08-14 10:00", periods=2, freq="min"))
    spot = pd.DataFrame({
        "timestamp": pd.date_range("2026-08-15 10:00", periods=2, freq="min"),
        "close": 500.0,
    })
    assert build_options_features(opts, spot).empty


def test_build_options_features_drops_expired_contracts():
    # Bars timestamped after the expiry must not produce features.
    ts = pd.date_range("2026-08-25 10:00", periods=2, freq="min")
    opts = _chain(ts)  # chain expires 2026-08-21
    spot = pd.DataFrame({"timestamp": ts, "close": 500.0})
    assert build_options_features(opts, spot).empty
