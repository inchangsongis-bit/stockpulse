"""
Generate realistic mock data for SPY so the full pipeline can run without API keys.
Uses geometric Brownian motion for price simulation and hand-crafted news articles.
"""

import random
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# OHLCV mock data — 3 years of DAILY bars for SPY
# (Minute bars for 3 years = ~14M rows — too heavy for SQLite prototype.
#  We generate daily bars + a recent week of minute bars for indicator testing.)
# ---------------------------------------------------------------------------

def generate_daily_ohlcv(
    ticker: str = "SPY",
    start_date: datetime = datetime(2023, 8, 1),
    end_date: datetime = datetime(2026, 8, 1),
    initial_price: float = 440.0,
    annual_drift: float = 0.10,
    annual_vol: float = 0.16,
) -> List[Dict[str, Any]]:
    """Generate realistic daily OHLCV bars using geometric Brownian motion."""
    random.seed(42)  # reproducible
    dt = 1 / 252  # trading day fraction
    daily_drift = annual_drift * dt
    daily_vol = annual_vol * math.sqrt(dt)

    bars = []
    price = initial_price
    current = start_date

    while current <= end_date:
        # skip weekends
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        # GBM step
        shock = random.gauss(0, 1)
        ret = daily_drift + daily_vol * shock
        close = price * math.exp(ret)

        # intraday range: high/low around open→close path
        intraday_range = abs(close - price) + price * random.uniform(0.002, 0.012)
        high = max(price, close) + random.uniform(0, intraday_range * 0.5)
        low = min(price, close) - random.uniform(0, intraday_range * 0.5)
        low = max(low, 1.0)  # floor

        # volume: base ~80M for SPY, with randomness and trend correlation
        base_vol = 80_000_000
        vol_mult = 1.0 + abs(ret) * 15 + random.uniform(-0.3, 0.3)
        volume = int(base_vol * vol_mult)

        vwap = (high + low + close) / 3

        bars.append({
            "ticker": ticker,
            "timestamp": current.replace(hour=16, minute=0),
            "open": round(price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": volume,
            "vwap": round(vwap, 2),
            "num_trades": int(volume / random.randint(80, 150)),
        })

        price = close
        current += timedelta(days=1)

    return bars


def generate_minute_ohlcv(
    ticker: str = "SPY",
    days: int = 5,
    base_date: datetime = datetime(2026, 8, 4),
    initial_price: float = 560.0,
) -> List[Dict[str, Any]]:
    """Generate minute-level OHLCV for the most recent N trading days."""
    random.seed(99)
    bars = []
    price = initial_price
    minutes_per_day = 390  # 9:30 AM to 4:00 PM

    for d in range(days):
        day = base_date + timedelta(days=d)
        if day.weekday() >= 5:
            continue

        market_open = day.replace(hour=9, minute=30, second=0)

        for m in range(minutes_per_day):
            ts = market_open + timedelta(minutes=m)

            # smaller moves at minute level
            shock = random.gauss(0, 1)
            ret = 0.00001 + 0.0008 * shock  # ~0.08% per minute vol
            close = price * math.exp(ret)

            spread = price * random.uniform(0.0001, 0.0005)
            high = max(price, close) + spread
            low = min(price, close) - spread
            low = max(low, 1.0)

            # volume varies by time of day (U-shape)
            hour_frac = m / minutes_per_day
            time_weight = 2.0 - 3.0 * hour_frac * (1 - hour_frac)  # higher at open/close
            volume = int(random.randint(50_000, 300_000) * time_weight)

            bars.append({
                "ticker": ticker,
                "timestamp": ts,
                "open": round(price, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(close, 4),
                "volume": volume,
                "vwap": round((high + low + close) / 3, 4),
                "num_trades": int(volume / random.randint(50, 120)),
            })

            price = close

    return bars


# ---------------------------------------------------------------------------
# Mock news articles
# ---------------------------------------------------------------------------

MOCK_NEWS: List[Dict[str, Any]] = [
    {
        "ticker": "SPY",
        "title": "Fed Signals Potential Rate Cut in September Meeting",
        "source": "Reuters",
        "url": "https://reuters.com/mock/fed-rate-cut",
        "summary": "Federal Reserve officials indicated openness to lowering interest rates at the upcoming September meeting, citing cooling inflation data and a softening labor market. Markets rallied on the news with the S&P 500 gaining 1.2%.",
        "category": "macro",
        "published_at": datetime(2026, 8, 7, 14, 30),
        "relevance": 0.95,
    },
    {
        "ticker": "SPY",
        "title": "Q2 Earnings Season Beats Expectations: 78% of S&P 500 Companies Top Estimates",
        "source": "Bloomberg",
        "url": "https://bloomberg.com/mock/q2-earnings",
        "summary": "With 90% of S&P 500 companies having reported, 78% beat earnings estimates, above the 5-year average of 73%. Revenue growth averaged 5.2% year-over-year, led by technology and healthcare sectors.",
        "category": "earnings",
        "published_at": datetime(2026, 8, 6, 10, 15),
        "relevance": 0.90,
    },
    {
        "ticker": "SPY",
        "title": "US Jobs Report Shows 180K Added, Unemployment Rises to 4.3%",
        "source": "CNBC",
        "url": "https://cnbc.com/mock/jobs-report",
        "summary": "The US economy added 180,000 jobs in July, below the expected 200,000. Unemployment ticked up to 4.3%, the highest since January 2022. Wage growth moderated to 3.6% annually.",
        "category": "macro",
        "published_at": datetime(2026, 8, 5, 8, 30),
        "relevance": 0.92,
    },
    {
        "ticker": "SPY",
        "title": "Treasury Yields Drop as Bond Market Prices In Rate Cuts",
        "source": "Wall Street Journal",
        "url": "https://wsj.com/mock/treasury-yields",
        "summary": "The 10-year Treasury yield fell to 3.85%, its lowest level since early 2024, as bond traders increased bets on multiple Fed rate cuts this year. The 2-year/10-year spread uninverted for the first time in over two years.",
        "category": "macro",
        "published_at": datetime(2026, 8, 8, 11, 0),
        "relevance": 0.88,
    },
    {
        "ticker": "SPY",
        "title": "China's Economic Slowdown Raises Global Growth Concerns",
        "source": "Financial Times",
        "url": "https://ft.com/mock/china-slowdown",
        "summary": "China's manufacturing PMI contracted for the third consecutive month, raising concerns about global demand. Multinational companies with significant China exposure saw selling pressure.",
        "category": "macro",
        "published_at": datetime(2026, 8, 4, 6, 0),
        "relevance": 0.72,
    },
    {
        "ticker": "SPY",
        "title": "AI Chip Demand Drives Tech Sector to New Highs",
        "source": "TechCrunch",
        "url": "https://techcrunch.com/mock/ai-chip-demand",
        "summary": "Semiconductor companies reported record demand for AI training and inference chips, with order backlogs extending into 2027. The technology sector now represents 32% of S&P 500 market cap.",
        "category": "sector",
        "published_at": datetime(2026, 8, 7, 9, 45),
        "relevance": 0.80,
    },
    {
        "ticker": "SPY",
        "title": "SEC Proposes New Rules for AI-Driven Trading Systems",
        "source": "SEC.gov",
        "url": "https://sec.gov/mock/ai-trading-rules",
        "summary": "The SEC proposed new disclosure requirements for firms using AI and machine learning in trading decisions, citing concerns about systemic risk from correlated algorithmic strategies.",
        "category": "regulatory",
        "published_at": datetime(2026, 8, 3, 16, 0),
        "relevance": 0.55,
    },
    {
        "ticker": "SPY",
        "title": "Consumer Spending Remains Resilient Despite Inflation Concerns",
        "source": "Associated Press",
        "url": "https://apnews.com/mock/consumer-spending",
        "summary": "US retail sales rose 0.4% in July, driven by back-to-school shopping and summer travel. Consumer confidence index held steady at 102, suggesting the economy is avoiding a hard landing.",
        "category": "macro",
        "published_at": datetime(2026, 8, 6, 13, 0),
        "relevance": 0.78,
    },
]


def get_mock_news(ticker: str = "SPY") -> List[Dict[str, Any]]:
    return [a for a in MOCK_NEWS if a["ticker"] == ticker]
