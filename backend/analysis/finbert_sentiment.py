"""
Free, local news sentiment scoring via FinBERT (ProsusAI/finbert) — a
BERT model fine-tuned specifically on financial text. No API calls, no
per-article cost, unlike the Claude-based scoring in
agents/sentiment_analyst.py.

The model (~400MB) is downloaded once on first use and cached by
huggingface_hub in the usual ~/.cache/huggingface location, then reused
across process runs. Loading is lazy (only on first score_text() call)
so importing this module — or starting the app — doesn't pay that cost
upfront.
"""

from functools import lru_cache
from typing import Tuple


@lru_cache
def _get_pipeline():
    from transformers import pipeline

    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


def score_text(text: str) -> Tuple[float, str, float]:
    """
    Score a piece of text's financial sentiment.

    Returns (sentiment, label, confidence):
      - sentiment: float in [-1.0, 1.0] — +confidence for "positive",
        -confidence for "negative", 0.0 for "neutral"
      - label: the raw FinBERT label ("positive" | "negative" | "neutral")
      - confidence: the model's confidence in that label, [0.0, 1.0]
    """
    # FinBERT's underlying model has a 512-token limit; truncating on
    # characters is a cheap, good-enough approximation that avoids
    # needing the tokenizer just to measure length.
    result = _get_pipeline()(text[:2000])[0]
    label = result["label"].lower()
    confidence = round(float(result["score"]), 3)

    if label == "positive":
        sentiment = confidence
    elif label == "negative":
        sentiment = -confidence
    else:
        sentiment = 0.0

    return sentiment, label, confidence
