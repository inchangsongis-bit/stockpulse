"""
Quantitative Analyst Agent
Computes technical indicators on OHLCV data and produces a TechnicalProfile.
"""

import json
from typing import Any, Dict
import pandas as pd
from agents.base import BaseAgent
from analysis.indicators import compute_all_indicators


class QuantAnalystAgent(BaseAgent):
    name = "quant_analyst"

    async def run(self, ticker: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.log(f"Computing technical indicators for {ticker}")

        ohlcv_data = context.get("ohlcv_data", [])
        if not ohlcv_data:
            self.log("No OHLCV data available")
            return self._empty_profile(ticker)

        df = pd.DataFrame(ohlcv_data)
        df = df.sort_values("timestamp").reset_index(drop=True)

        indicators = compute_all_indicators(df)

        profile = {
            "ticker": ticker,
            "timestamp": df["timestamp"].iloc[-1].isoformat() if hasattr(df["timestamp"].iloc[-1], "isoformat") else str(df["timestamp"].iloc[-1]),
            "trend_score": indicators["trend_score"],
            "momentum_score": indicators["momentum_score"],
            "volatility_state": indicators["volatility_state"],
            "volume_anomaly": indicators["volume_anomaly"],
            "volume_anomaly_magnitude": indicators["volume_anomaly_magnitude"],
            "support": indicators["support"],
            "resistance": indicators["resistance"],
            "patterns": indicators["patterns"],
            "indicators": indicators["indicators"],
        }

        self.log(f"Trend: {profile['trend_score']}, Momentum: {profile['momentum_score']}, "
                 f"Volatility: {profile['volatility_state']}")

        return profile

    def _empty_profile(self, ticker: str) -> Dict[str, Any]:
        return {
            "ticker": ticker,
            "timestamp": None,
            "trend_score": 0.0,
            "momentum_score": 0.0,
            "volatility_state": "unknown",
            "volume_anomaly": False,
            "volume_anomaly_magnitude": 1.0,
            "support": 0.0,
            "resistance": 0.0,
            "patterns": [],
            "indicators": {},
        }
