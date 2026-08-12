"""
Pipeline Orchestrator
Sequences agents: News Researcher → Quant Analyst → Sentiment Analyst → Strategy Engine.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.news_researcher import NewsResearcherAgent
from agents.quant_analyst import QuantAnalystAgent
from agents.sentiment_analyst import SentimentAnalystAgent
from agents.strategy_engine import StrategyEngineAgent


class PipelineOrchestrator:
    def __init__(self):
        self.news_researcher = NewsResearcherAgent()
        self.quant_analyst = QuantAnalystAgent()
        self.sentiment_analyst = SentimentAnalystAgent()
        self.strategy_engine = StrategyEngineAgent()

    async def run_pipeline(
        self,
        ticker: str,
        ohlcv_data: List[Dict[str, Any]],
        use_mock: bool = True,
        cached_sentiment: Optional[Dict[str, dict]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full analysis pipeline for a single ticker.
        Returns the complete pipeline output including all intermediate results.
        """
        started = datetime.now()
        print(f"\n{'='*60}")
        print(f"  PIPELINE START: {ticker} at {started.strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")

        results: Dict[str, Any] = {"ticker": ticker, "started": started.isoformat()}

        # Step 1: News Researcher
        print("── Step 1/4: News Research ──")
        research = await self.news_researcher.run(ticker, {"use_mock": use_mock})
        results["research"] = research

        # Step 2: Quant Analyst (runs on OHLCV data, independent of news)
        print("\n── Step 2/4: Technical Analysis ──")
        technical_profile = await self.quant_analyst.run(ticker, {"ohlcv_data": ohlcv_data})
        results["technical_profile"] = technical_profile

        # Step 3: Sentiment Analyst (needs research output)
        print("\n── Step 3/4: Sentiment Analysis ──")
        sentiment_profile = await self.sentiment_analyst.run(
            ticker, {"research": research, "cached_sentiment": cached_sentiment or {}}
        )
        results["sentiment_profile"] = sentiment_profile

        # Step 4: Strategy Engine (needs technical + sentiment)
        print("\n── Step 4/4: Strategy Signal Generation ──")
        signal = await self.strategy_engine.run(
            ticker,
            {
                "technical_profile": technical_profile,
                "sentiment_profile": sentiment_profile,
            },
        )
        results["signal"] = signal

        elapsed = (datetime.now() - started).total_seconds()
        results["elapsed_seconds"] = round(elapsed, 2)

        print(f"\n{'='*60}")
        print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
        print(f"  Signal: {signal['action']} | Confidence: {signal['confidence']}%")
        print(f"{'='*60}\n")

        return results
