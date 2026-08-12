"""
Sentiment Analyst Agent
Uses Claude API to analyze news sentiment, source credibility, and expected impact.
Falls back to rule-based scoring if no API key is configured.
"""

import json
from typing import Any, Dict, List
from agents.base import BaseAgent, extract_claude_text
from config import get_settings


class SentimentAnalystAgent(BaseAgent):
    name = "sentiment_analyst"

    async def run(self, ticker: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.log(f"Analyzing sentiment for {ticker}")

        research = context.get("research", {})
        articles = research.get("articles", [])

        if not articles:
            self.log("No articles to analyze")
            return self._empty_profile(ticker)

        settings = get_settings()

        # Skip re-scoring articles we already have a cached sentiment score
        # for (keyed by URL) — relevance is always refreshed from the
        # current fetch since it's a function of *this run's* recency
        # window, not a static property of the article.
        cached_sentiment: Dict[str, dict] = context.get("cached_sentiment", {})
        cached_scores, to_score = [], []
        for a in articles:
            cached = cached_sentiment.get(a.get("url"))
            if cached:
                cached_scores.append({**cached, "relevance": a["relevance"]})
            else:
                to_score.append(a)

        if cached_scores:
            self.log(f"Using cached sentiment for {len(cached_scores)} articles, scoring {len(to_score)} new")

        if not to_score:
            new_scores = []
        elif settings.anthropic_api_key:
            new_scores = await self._claude_sentiment(to_score, ticker)
        else:
            self.log("No Anthropic API key — using rule-based sentiment")
            new_scores = self._rule_based_sentiment(to_score)

        article_scores = cached_scores + new_scores

        # Aggregate: weighted by relevance and recency
        if article_scores:
            total_weight = sum(s["relevance"] for s in article_scores)
            if total_weight > 0:
                composite = sum(s["sentiment"] * s["relevance"] for s in article_scores) / total_weight
            else:
                composite = 0.0
        else:
            composite = 0.0

        # Determine trend from article distribution
        positive = sum(1 for s in article_scores if s["sentiment"] > 0.1)
        negative = sum(1 for s in article_scores if s["sentiment"] < -0.1)
        if positive > negative * 1.5:
            trend = "improving"
        elif negative > positive * 1.5:
            trend = "declining"
        else:
            trend = "stable"

        profile = {
            "ticker": ticker,
            "timestamp": research.get("date"),
            "composite_sentiment": round(composite, 3),
            "insider_signal": "neutral",  # placeholder until EDGAR integration
            "sentiment_trend": trend,
            "article_scores": article_scores,
        }

        self.log(f"Composite sentiment: {profile['composite_sentiment']}, Trend: {trend}")
        return profile

    async def _claude_sentiment(self, articles: List[Dict], ticker: str) -> List[Dict]:
        """Use Claude to score each article's sentiment and impact."""
        import anthropic

        client = anthropic.Anthropic(api_key=get_settings().anthropic_api_key)

        articles_text = "\n\n".join(
            f"Article {i+1}: [{a['source']}] {a['title']}\n{a['summary']}"
            for i, a in enumerate(articles)
        )

        prompt = f"""Analyze the sentiment of each news article below as it relates to {ticker} stock price.

For each article, return a JSON array with objects containing:
- "index": article number (1-based)
- "sentiment": float from -1.0 (very bearish) to +1.0 (very bullish)
- "source_credibility": float from 0.0 to 1.0 (how reliable is this source)
- "expected_impact": "none" | "low" | "medium" | "high"
- "reasoning": one sentence explaining your score

Articles:
{articles_text}

Return ONLY the JSON array, no other text."""

        import anthropic as anthropic_module

        try:
            response = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic_module.APIError as e:
            self.log(f"Claude API error: {e}, falling back to rules")
            return self._rule_based_sentiment(articles)

        try:
            scores_raw = json.loads(extract_claude_text(response))
            result = []
            for score in scores_raw:
                idx = score["index"] - 1
                if 0 <= idx < len(articles):
                    result.append({
                        "title": articles[idx]["title"],
                        "source": articles[idx]["source"],
                        "url": articles[idx].get("url", ""),
                        "sentiment": float(score["sentiment"]),
                        "source_credibility": float(score["source_credibility"]),
                        "expected_impact": score["expected_impact"],
                        "reasoning": score.get("reasoning", ""),
                        "relevance": articles[idx]["relevance"],
                    })
            return result
        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            self.log(f"Claude response parse error: {e}, falling back to rules")
            return self._rule_based_sentiment(articles)

    def _rule_based_sentiment(self, articles: List[Dict]) -> List[Dict]:
        """Simple keyword-based sentiment scoring as fallback."""
        bullish_words = {
            "beat", "beats", "exceeded", "rally", "rallied", "gain", "gains",
            "growth", "bullish", "upgrade", "resilient", "strong", "surge",
            "record", "high", "positive", "optimistic", "cut rate", "rate cut",
        }
        bearish_words = {
            "miss", "missed", "below", "decline", "drop", "fell", "loss",
            "bearish", "downgrade", "weak", "concern", "risk", "slowdown",
            "contraction", "recession", "inflation", "unemployment rises",
        }

        # Source credibility lookup
        credibility = {
            "Reuters": 0.95, "Bloomberg": 0.95, "Wall Street Journal": 0.92,
            "CNBC": 0.85, "Financial Times": 0.92, "Associated Press": 0.90,
            "SEC.gov": 0.98, "TechCrunch": 0.70,
        }

        results = []
        for a in articles:
            text = (a.get("title", "") + " " + a.get("summary", "")).lower()
            bull = sum(1 for w in bullish_words if w in text)
            bear = sum(1 for w in bearish_words if w in text)
            total = bull + bear
            if total == 0:
                sentiment = 0.0
            else:
                sentiment = (bull - bear) / total
                sentiment = max(min(sentiment, 1.0), -1.0)

            source_cred = credibility.get(a.get("source", ""), 0.5)

            # Impact based on relevance
            rel = a.get("relevance", 0.5)
            if rel > 0.85:
                impact = "high"
            elif rel > 0.6:
                impact = "medium"
            else:
                impact = "low"

            results.append({
                "title": a["title"],
                "source": a.get("source", "unknown"),
                "url": a.get("url", ""),
                "sentiment": round(sentiment, 3),
                "source_credibility": source_cred,
                "expected_impact": impact,
                "reasoning": f"Keyword analysis: {bull} bullish, {bear} bearish signals",
                "relevance": a.get("relevance", 0.5),
            })

        return results

    def _empty_profile(self, ticker: str) -> Dict[str, Any]:
        return {
            "ticker": ticker,
            "timestamp": None,
            "composite_sentiment": 0.0,
            "insider_signal": "neutral",
            "sentiment_trend": "stable",
            "article_scores": [],
        }
