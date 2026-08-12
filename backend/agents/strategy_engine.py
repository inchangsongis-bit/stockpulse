"""
Strategy Engine Agent
Synthesizes TechnicalProfile + SentimentProfile into actionable signals.
This is the final decision layer.
"""

import json
from typing import Any, Dict
from agents.base import BaseAgent, extract_claude_text
from config import get_settings


class StrategyEngineAgent(BaseAgent):
    name = "strategy_engine"

    # Weights for composite scoring
    WEIGHTS = {
        "technical": 0.40,
        "sentiment": 0.30,
        "volume": 0.15,
        "momentum": 0.15,
    }

    async def run(self, ticker: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.log(f"Generating signal for {ticker}")

        technical = context.get("technical_profile", {})
        sentiment = context.get("sentiment_profile", {})

        # Extract scores
        trend_score = technical.get("trend_score", 0.0)
        momentum_score = technical.get("momentum_score", 0.0)
        sentiment_score = sentiment.get("composite_sentiment", 0.0)
        vol_anomaly = technical.get("volume_anomaly", False)
        vol_magnitude = technical.get("volume_anomaly_magnitude", 1.0)
        indicators = technical.get("indicators", {})
        current_price = indicators.get("current_price", 0.0)

        # Volume score: anomalous volume in trend direction is confirming
        if vol_anomaly:
            vol_direction = 1 if trend_score > 0 else -1
            volume_score = vol_direction * min(vol_magnitude / 3, 1.0)
        else:
            volume_score = 0.0

        # Weighted composite
        composite = (
            self.WEIGHTS["technical"] * trend_score
            + self.WEIGHTS["sentiment"] * sentiment_score
            + self.WEIGHTS["volume"] * volume_score
            + self.WEIGHTS["momentum"] * momentum_score
        )

        # Determine action
        if composite > 0.25:
            action = "BUY"
        elif composite < -0.25:
            action = "SELL"
        else:
            action = "HOLD"

        # Conflicting signals → force HOLD
        tech_direction = "bullish" if trend_score > 0.15 else ("bearish" if trend_score < -0.15 else "neutral")
        sent_direction = "bullish" if sentiment_score > 0.15 else ("bearish" if sentiment_score < -0.15 else "neutral")

        if tech_direction == "bullish" and sent_direction == "bearish":
            action = "HOLD"
            conflict_note = "Conflicting signals: bullish technicals vs bearish sentiment."
        elif tech_direction == "bearish" and sent_direction == "bullish":
            action = "HOLD"
            conflict_note = "Conflicting signals: bearish technicals vs bullish sentiment."
        else:
            conflict_note = ""

        # Confidence (0-100, capped at 88 — markets are uncertain)
        raw_confidence = abs(composite) * 100
        confidence = min(int(raw_confidence), 88)

        # Entry range, target, stop loss
        atr_val = indicators.get("atr_14", current_price * 0.01)
        support = technical.get("support", current_price * 0.95)
        resistance = technical.get("resistance", current_price * 1.05)

        if action == "BUY":
            entry_low = round(current_price - atr_val * 0.3, 2)
            entry_high = round(current_price + atr_val * 0.2, 2)
            target = round(resistance * 1.02, 2)
            stop_loss = round(support * 0.98, 2)
        elif action == "SELL":
            entry_low = round(current_price - atr_val * 0.2, 2)
            entry_high = round(current_price + atr_val * 0.3, 2)
            target = round(support * 0.98, 2)
            stop_loss = round(resistance * 1.02, 2)
        else:
            entry_low = entry_high = current_price
            target = current_price
            stop_loss = current_price

        # Time horizon based on volatility
        vol_state = technical.get("volatility_state", "unknown")
        if vol_state == "expanding":
            time_horizon = "1-2 weeks"
        else:
            time_horizon = "2-4 weeks"

        # Risk level
        if atr_val / current_price > 0.03:
            risk_level = "high"
        elif atr_val / current_price > 0.015:
            risk_level = "moderate"
        else:
            risk_level = "low"

        # Build reasoning
        reasoning = await self._build_reasoning(
            ticker, action, confidence, technical, sentiment, conflict_note
        )

        signal = {
            "ticker": ticker,
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
            "entry_low": entry_low,
            "entry_high": entry_high,
            "target": target,
            "stop_loss": stop_loss,
            "time_horizon": time_horizon,
            "risk_level": risk_level,
            "composite_score": round(composite, 3),
            "factors": {
                "technical": {
                    "score": round(trend_score, 3),
                    "weight": self.WEIGHTS["technical"],
                    "weighted": round(trend_score * self.WEIGHTS["technical"], 3),
                },
                "sentiment": {
                    "score": round(sentiment_score, 3),
                    "weight": self.WEIGHTS["sentiment"],
                    "weighted": round(sentiment_score * self.WEIGHTS["sentiment"], 3),
                },
                "volume": {
                    "score": round(volume_score, 3),
                    "weight": self.WEIGHTS["volume"],
                    "weighted": round(volume_score * self.WEIGHTS["volume"], 3),
                },
                "momentum": {
                    "score": round(momentum_score, 3),
                    "weight": self.WEIGHTS["momentum"],
                    "weighted": round(momentum_score * self.WEIGHTS["momentum"], 3),
                },
            },
            "disclaimer": "For informational purposes only. Not financial advice.",
        }

        self.log(f"Signal: {action} (confidence {confidence}%)")
        return signal

    async def _build_reasoning(
        self, ticker, action, confidence, technical, sentiment, conflict_note
    ) -> str:
        """Use Claude API for natural-language reasoning if available, else template."""
        settings = get_settings()

        if settings.anthropic_api_key:
            try:
                return await self._claude_reasoning(
                    ticker, action, confidence, technical, sentiment, conflict_note
                )
            except Exception as e:
                import anthropic

                if isinstance(e, (anthropic.APIError, ValueError)):
                    self.log(f"Claude reasoning failed: {e}, falling back to template reasoning")
                else:
                    raise

        return self._template_reasoning(ticker, action, confidence, technical, sentiment, conflict_note)

    def _template_reasoning(
        self, ticker, action, confidence, technical, sentiment, conflict_note
    ) -> str:
        indicators = technical.get("indicators", {})
        parts = [f"Signal: {action} with {confidence}% confidence for {ticker}."]

        if conflict_note:
            parts.append(conflict_note)

        parts.append(
            f"Technical: RSI at {indicators.get('rsi_14', 'N/A')}, "
            f"MACD histogram {'positive' if indicators.get('macd_histogram', 0) > 0 else 'negative'}, "
            f"price {'above' if technical.get('trend_score', 0) > 0 else 'below'} key moving averages."
        )
        parts.append(
            f"Sentiment: composite score {sentiment.get('composite_sentiment', 0):.2f}, "
            f"trend {sentiment.get('sentiment_trend', 'stable')}."
        )

        patterns = technical.get("patterns", [])
        if patterns:
            parts.append(f"Patterns detected: {', '.join(patterns)}.")

        return " ".join(parts)

    async def _claude_reasoning(
        self, ticker, action, confidence, technical, sentiment, conflict_note
    ) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=get_settings().anthropic_api_key)

        data_summary = json.dumps({
            "action": action,
            "confidence": confidence,
            "technical": {
                "trend_score": technical.get("trend_score"),
                "momentum_score": technical.get("momentum_score"),
                "volatility_state": technical.get("volatility_state"),
                "patterns": technical.get("patterns"),
                "indicators": technical.get("indicators"),
            },
            "sentiment": {
                "composite": sentiment.get("composite_sentiment"),
                "trend": sentiment.get("sentiment_trend"),
            },
            "conflict": conflict_note,
        }, indent=2)

        prompt = f"""Based on the following analysis data for {ticker}, write a concise 2-3 sentence reasoning
for the {action} signal with {confidence}% confidence. Be specific about which indicators
and news factors drove the decision. Do NOT give financial advice — describe the analysis objectively.

{data_summary}

Write the reasoning as a single paragraph, no bullet points."""

        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        return extract_claude_text(response)
