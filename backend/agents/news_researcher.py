"""
News Researcher Agent
Fetches news from APIs (or mock data) and produces structured research briefs.
"""

from typing import Any, Dict, List
from datetime import datetime
from agents.base import BaseAgent
from mock_data import get_mock_news


class NewsResearcherAgent(BaseAgent):
    name = "news_researcher"

    async def run(self, ticker: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.log(f"Researching news for {ticker}")

        # Use mock data for now; swap to API calls when keys are available
        use_mock = context.get("use_mock", True)

        if use_mock:
            articles = get_mock_news(ticker)
        else:
            articles = await self._fetch_live_news(ticker)

        self.log(f"Found {len(articles)} articles for {ticker}")

        return {
            "ticker": ticker,
            "date": datetime.now().isoformat(),
            "article_count": len(articles),
            "articles": [
                {
                    "title": a["title"],
                    "source": a["source"],
                    "url": a["url"],
                    "summary": a["summary"],
                    "category": a["category"],
                    "published_at": a["published_at"].isoformat() if isinstance(a["published_at"], datetime) else a["published_at"],
                    "relevance": a["relevance"],
                }
                for a in articles
            ],
        }

    async def _fetch_live_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Placeholder for live API calls (Finnhub, Alpha Vantage)."""
        # TODO: Implement when API keys are provided
        # import httpx
        # async with httpx.AsyncClient() as client:
        #     resp = await client.get(f"https://finnhub.io/api/v1/company-news", params={...})
        return []
