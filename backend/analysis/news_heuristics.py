"""
Small shared heuristics for scoring news articles, used by both the
rule-based sentiment fallback (agents/sentiment_analyst.py) and the
FinBERT sentiment path (analysis/finbert_sentiment.py's caller in
routers/stocks.py) — pulled out so the two don't drift out of sync.
"""

_SOURCE_CREDIBILITY = {
    "Reuters": 0.95, "Bloomberg": 0.95, "Wall Street Journal": 0.92,
    "CNBC": 0.85, "Financial Times": 0.92, "Associated Press": 0.90,
    "SEC.gov": 0.98, "TechCrunch": 0.70,
}


def source_credibility(source: str) -> float:
    return _SOURCE_CREDIBILITY.get(source, 0.5)


def impact_from_relevance(relevance: float) -> str:
    if relevance > 0.85:
        return "high"
    elif relevance > 0.6:
        return "medium"
    return "low"
