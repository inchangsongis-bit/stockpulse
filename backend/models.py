from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, BigInteger
from sqlalchemy.sql import func
from database import Base


class OHLCV(Base):
    __tablename__ = "ohlcv"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    interval = Column(String(10), nullable=False, default="daily", server_default="daily", index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)
    vwap = Column(Float)
    num_trades = Column(Integer)


class Watchlist(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, unique=True)
    added_at = Column(DateTime, server_default=func.now())


class NewsArticle(Base):
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    source = Column(String(100))
    url = Column(String(1000))
    summary = Column(Text)
    category = Column(String(50))
    published_at = Column(DateTime)
    relevance = Column(Float)
    fetched_at = Column(DateTime, server_default=func.now())


class TechnicalProfile(Base):
    __tablename__ = "technical_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    trend_score = Column(Float)
    momentum_score = Column(Float)
    volatility_state = Column(String(20))
    volume_anomaly = Column(Boolean, default=False)
    volume_anomaly_magnitude = Column(Float)
    support_level = Column(Float)
    resistance_level = Column(Float)
    patterns_detected = Column(Text)  # JSON string
    indicators_json = Column(Text)  # JSON string of raw indicator values


class SentimentProfile(Base):
    __tablename__ = "sentiment_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    composite_sentiment = Column(Float)
    insider_signal = Column(String(20))
    sentiment_trend = Column(String(20))
    article_scores_json = Column(Text)  # JSON string


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    action = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Integer)
    reasoning = Column(Text)
    entry_low = Column(Float)
    entry_high = Column(Float)
    target = Column(Float)
    stop_loss = Column(Float)
    time_horizon = Column(String(50))
    risk_level = Column(String(20))
    factors_json = Column(Text)  # JSON string
