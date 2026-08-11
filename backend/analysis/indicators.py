"""
Technical indicator calculations on pandas DataFrames.
All functions expect a DataFrame with columns: open, high, low, close, volume
indexed or containing a 'timestamp' column, sorted ascending.
"""

import pandas as pd
import numpy as np


# ── Trend Indicators ──────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — trend strength (0-100)."""
    high, low, close = df["high"], df["low"], df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = _true_range(df)
    atr_val = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_val)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0)
    return dx.rolling(window=period).mean()


# ── Momentum Indicators ──────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0-100)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    # No losses in the window (all gains) is maximally overbought, not undefined.
    result = result.where(avg_loss != 0, 100.0)
    # No gains and no losses (flat price) is neutral, not undefined.
    result = result.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return result


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """Returns (%K, %D)."""
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return k, d


def rate_of_change(close: pd.Series, period: int = 12) -> pd.Series:
    return ((close - close.shift(period)) / close.shift(period)) * 100


# ── Volatility Indicators ────────────────────────────────────────────────

def _true_range(df: pd.DataFrame) -> pd.Series:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    return _true_range(df).rolling(window=period).mean()


def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Returns (upper, middle, lower, bandwidth)."""
    middle = sma(close, period)
    rolling_std = close.rolling(window=period).std()
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    bandwidth = ((upper - lower) / middle) * 100
    return upper, middle, lower, bandwidth


# ── Volume Indicators ────────────────────────────────────────────────────

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price (intraday — resets each day)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def volume_sma_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Current volume vs its SMA — >1.5 is anomalous."""
    vol_sma = sma(volume, period)
    return volume / vol_sma.replace(0, np.nan)


# ── Composite Calculations ───────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame) -> dict:
    """
    Compute all indicators on a DataFrame and return the latest values
    plus trend/momentum scores.
    """
    close = df["close"]
    n = len(df)

    # Need at least 200 bars for SMA200
    if n < 200:
        # Fallback: compute what we can
        pass

    # Individual indicators
    rsi_val = rsi(close).iloc[-1] if n >= 14 else 50.0
    macd_line, macd_signal, macd_hist = macd(close)
    macd_val = macd_hist.iloc[-1] if n >= 26 else 0.0

    sma_20 = sma(close, 20).iloc[-1] if n >= 20 else close.iloc[-1]
    sma_50 = sma(close, 50).iloc[-1] if n >= 50 else close.iloc[-1]
    sma_200 = sma(close, 200).iloc[-1] if n >= 200 else close.iloc[-1]

    bb_upper, bb_mid, bb_lower, bb_bw = bollinger_bands(close)
    bb_bw_val = bb_bw.iloc[-1] if n >= 20 else 0.0

    atr_val = atr(df).iloc[-1] if n >= 14 else 0.0
    obv_val = obv(close, df["volume"]).iloc[-1]
    vol_ratio = volume_sma_ratio(df["volume"]).iloc[-1] if n >= 20 else 1.0

    stoch_k, stoch_d = stochastic(df)
    stoch_k_val = stoch_k.iloc[-1] if n >= 14 else 50.0

    adx_val = adx(df).iloc[-1] if n >= 28 else 25.0
    roc_val = rate_of_change(close).iloc[-1] if n >= 12 else 0.0

    current_price = close.iloc[-1]

    # ── Trend score (-1 to +1) ──
    trend_signals = []
    # Price vs SMAs
    if n >= 20:
        trend_signals.append(1 if current_price > sma_20 else -1)
    if n >= 50:
        trend_signals.append(1 if current_price > sma_50 else -1)
    if n >= 200:
        trend_signals.append(1 if current_price > sma_200 else -1)
    # SMA crossovers
    if n >= 50:
        trend_signals.append(1 if sma_20 > sma_50 else -1)
    # MACD
    trend_signals.append(1 if macd_val > 0 else -1)

    trend_score = sum(trend_signals) / max(len(trend_signals), 1)

    # ── Momentum score (-1 to +1) ──
    mom_signals = []
    # RSI
    if rsi_val > 70:
        mom_signals.append(-0.5)  # overbought, bearish
    elif rsi_val < 30:
        mom_signals.append(0.5)   # oversold, bullish
    else:
        mom_signals.append((rsi_val - 50) / 50)
    # Stochastic
    if stoch_k_val > 80:
        mom_signals.append(-0.3)
    elif stoch_k_val < 20:
        mom_signals.append(0.3)
    else:
        mom_signals.append((stoch_k_val - 50) / 100)
    # ROC
    mom_signals.append(max(min(roc_val / 10, 1), -1))

    momentum_score = sum(mom_signals) / max(len(mom_signals), 1)

    # ── Volatility state ──
    if n >= 40:
        recent_bw = bb_bw.iloc[-5:].mean() if n >= 25 else bb_bw_val
        older_bw = bb_bw.iloc[-20:-5].mean() if n >= 40 else recent_bw
        volatility_state = "expanding" if recent_bw > older_bw else "contracting"
    else:
        volatility_state = "unknown"

    # ── Volume anomaly ──
    volume_anomaly = bool(vol_ratio > 1.5)

    # ── Support / Resistance (simple: recent low/high) ──
    lookback = min(60, n)
    support = float(df["low"].iloc[-lookback:].min())
    resistance = float(df["high"].iloc[-lookback:].max())

    # ── Pattern detection (simplified) ──
    patterns = []
    if n >= 3:
        c = close.iloc[-3:].tolist()
        if c[2] > c[1] > c[0]:
            patterns.append("three_white_soldiers" if (c[2] - c[0]) / c[0] > 0.02 else "bullish_trend")
        elif c[2] < c[1] < c[0]:
            patterns.append("three_black_crows" if (c[0] - c[2]) / c[0] > 0.02 else "bearish_trend")
    if n >= 20 and current_price > resistance * 0.99:
        patterns.append("resistance_test")
    if n >= 20 and current_price < support * 1.01:
        patterns.append("support_test")

    return {
        "trend_score": round(float(trend_score), 3),
        "momentum_score": round(float(momentum_score), 3),
        "volatility_state": volatility_state,
        "volume_anomaly": volume_anomaly,
        "volume_anomaly_magnitude": round(float(vol_ratio), 2),
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "patterns": patterns,
        "indicators": {
            "rsi_14": round(float(rsi_val), 2),
            "macd_histogram": round(float(macd_val), 4),
            "sma_20": round(float(sma_20), 2),
            "sma_50": round(float(sma_50), 2),
            "sma_200": round(float(sma_200), 2),
            "bollinger_bandwidth": round(float(bb_bw_val), 2),
            "atr_14": round(float(atr_val), 2),
            "obv": round(float(obv_val), 0),
            "volume_sma_ratio": round(float(vol_ratio), 2),
            "stochastic_k": round(float(stoch_k_val), 2),
            "adx_14": round(float(adx_val), 2),
            "roc_12": round(float(roc_val), 2),
            "current_price": round(float(current_price), 2),
        },
    }
