"""
Options-derived features for SPY, built from the pilot dataset collected
by scripts/fetch_options_pilot.py.

Polygon's greeks and open-interest endpoints are 403 on this tier, so
implied volatility is recovered here by inverting Black-Scholes from data
we DO have: the option's mid price, its strike and expiry, and the
underlying's price at the same minute. Open interest has no such
workaround, which is why gamma exposure — the most-cited options-derived
intraday signal — is not computable and is not attempted.

Features produced, all contemporaneous (known at time t, used to predict
t+5, so no lookahead):

    put_call_volume_ratio   put contract volume / call contract volume
                            across the near-the-money chain this minute.
    atm_iv                  volume-weighted implied volatility of the
                            nearest-the-money contracts.
    iv_skew                 put IV minus call IV at comparable moneyness —
                            how much more the market is paying for
                            downside protection.
    options_volume_ratio    total near-money option volume this minute
                            against its own 20-minute average.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

# Short-dated US rate over the pilot window. IV is not very sensitive to
# this for near-the-money, near-dated contracts.
RISK_FREE_RATE = 0.04
MIN_PRICE = 0.05          # below this, option quotes are mostly noise
MAX_IV = 5.0              # inversion bound


def _bs_price(spot, strike, tau, rate, sigma, opt_type):
    if tau <= 0 or sigma <= 0:
        return max(0.0, (spot - strike) if opt_type == "call" else (strike - spot))
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if opt_type == "call":
        return spot * norm.cdf(d1) - strike * np.exp(-rate * tau) * norm.cdf(d2)
    return strike * np.exp(-rate * tau) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def implied_vol(price, spot, strike, tau, opt_type, rate=RISK_FREE_RATE):
    """Invert Black-Scholes for sigma. Returns NaN when the price is below
    intrinsic value or outside the solvable range — both common for thin
    contracts, and better left missing than silently clamped."""
    if not np.isfinite(price) or price < MIN_PRICE or tau <= 0:
        return np.nan
    intrinsic = max(0.0, (spot - strike) if opt_type == "call" else (strike - spot))
    if price <= intrinsic:
        return np.nan

    def objective(sigma):
        return _bs_price(spot, strike, tau, rate, sigma, opt_type) - price

    try:
        if objective(1e-6) * objective(MAX_IV) > 0:
            return np.nan
        return brentq(objective, 1e-6, MAX_IV, maxiter=60, xtol=1e-6)
    except (ValueError, RuntimeError):
        return np.nan


def build_options_features(option_bars: pd.DataFrame, spot_bars: pd.DataFrame) -> pd.DataFrame:
    """
    option_bars: contract, expiration, strike, opt_type, timestamp, close, volume
    spot_bars:   timestamp, close  (the underlying, same minute grid)

    Returns one row per minute with the features described above.
    """
    df = option_bars.merge(
        spot_bars.rename(columns={"close": "spot"})[["timestamp", "spot"]],
        on="timestamp", how="inner",
    )
    if df.empty:
        return pd.DataFrame()

    df["tau"] = (
        pd.to_datetime(df["expiration"]) + pd.Timedelta(hours=13)
        - pd.to_datetime(df["timestamp"])
    ).dt.total_seconds() / (365.25 * 24 * 3600)
    df = df[df["tau"] > 0]
    df["moneyness"] = (df["strike"] - df["spot"]).abs() / df["spot"]

    # ---- put/call volume ratio and overall activity ----
    vol = df.pivot_table(index="timestamp", columns="opt_type", values="volume",
                         aggfunc="sum").fillna(0.0)
    for col in ("call", "put"):
        if col not in vol.columns:
            vol[col] = 0.0
    out = pd.DataFrame(index=vol.index)
    out["put_call_volume_ratio"] = vol["put"] / vol["call"].replace(0, np.nan)
    total_vol = vol["call"] + vol["put"]
    out["options_volume_ratio"] = total_vol / total_vol.rolling(20, min_periods=20).mean().replace(0, np.nan)

    # ---- implied volatility, on the nearest-the-money contracts only ----
    # Inversion is the expensive step, so restrict it to contracts close to
    # the money, where IV is meaningful and the numerics are well behaved.
    atm = df[(df["moneyness"] <= 0.01) & (df["close"] >= MIN_PRICE) & (df["volume"] > 0)].copy()
    if not atm.empty:
        atm["iv"] = [
            implied_vol(p, s, k, t, o)
            for p, s, k, t, o in zip(atm["close"], atm["spot"], atm["strike"],
                                     atm["tau"], atm["opt_type"])
        ]
        atm = atm[atm["iv"].notna()]

    if not atm.empty:
        atm["wv"] = atm["iv"] * atm["volume"]
        g = atm.groupby("timestamp")
        out["atm_iv"] = g["wv"].sum() / g["volume"].sum().replace(0, np.nan)

        by_type = atm.pivot_table(index="timestamp", columns="opt_type", values="iv", aggfunc="mean")
        if {"put", "call"} <= set(by_type.columns):
            out["iv_skew"] = by_type["put"] - by_type["call"]

    for col in ("atm_iv", "iv_skew"):
        if col not in out.columns:
            out[col] = np.nan

    return out.reset_index()


OPTIONS_FEATURE_COLUMNS = [
    "put_call_volume_ratio",
    "atm_iv",
    "iv_skew",
    "options_volume_ratio",
]
