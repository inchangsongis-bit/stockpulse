from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from config import get_settings

engine = create_async_engine(get_settings().database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def _add_missing_columns(sync_conn):
    """
    Lightweight forward migration for the dev SQLite DB: add columns that
    exist on the model but not yet on disk. There's no Alembic set up for
    this prototype, so this keeps `create_all` (which only creates missing
    tables, never alters existing ones) from leaving stale schemas behind.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(sync_conn)
    migrations = [
        ("ohlcv", "interval", "ALTER TABLE ohlcv ADD COLUMN interval VARCHAR(10) NOT NULL DEFAULT 'daily'"),
        ("news_articles", "external_id", "ALTER TABLE news_articles ADD COLUMN external_id INTEGER"),
        ("news_articles", "sentiment", "ALTER TABLE news_articles ADD COLUMN sentiment FLOAT"),
        ("news_articles", "source_credibility", "ALTER TABLE news_articles ADD COLUMN source_credibility FLOAT"),
        ("news_articles", "expected_impact", "ALTER TABLE news_articles ADD COLUMN expected_impact VARCHAR(10)"),
        ("news_articles", "reasoning", "ALTER TABLE news_articles ADD COLUMN reasoning TEXT"),
        ("news_articles", "sentiment_scored_at", "ALTER TABLE news_articles ADD COLUMN sentiment_scored_at DATETIME"),
    ]
    table_names = inspector.get_table_names()
    for table, column, ddl in migrations:
        if table not in table_names:
            continue
        existing = {c["name"] for c in inspector.get_columns(table)}
        if column not in existing:
            sync_conn.execute(text(ddl))


def _add_missing_indexes(sync_conn):
    """
    Same idea as _add_missing_columns, for indexes: create_all only builds
    a table's indexes when it creates the table itself, so an index added
    to an existing model never lands on an already-populated dev DB.

    The composite (ticker, timestamp) index matters a lot now that the
    minute-bar backfill pushed ohlcv past 13M rows. Without it, "latest
    bar per ticker" makes SQLite sort each ticker's rows from scratch
    ("USE TEMP B-TREE FOR LAST TERM OF ORDER BY"), which is what turned
    the watchlist summary endpoint into a multi-second request.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(sync_conn)
    index_migrations = [
        ("ohlcv", "ix_ohlcv_ticker_timestamp",
         "CREATE INDEX IF NOT EXISTS ix_ohlcv_ticker_timestamp ON ohlcv (ticker, timestamp)"),
    ]
    table_names = inspector.get_table_names()
    for table, index_name, ddl in index_migrations:
        if table not in table_names:
            continue
        existing = {ix["name"] for ix in inspector.get_indexes(table)}
        if index_name not in existing:
            sync_conn.execute(text(ddl))


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)
        await conn.run_sync(_add_missing_indexes)


async def get_db():
    async with async_session() as session:
        yield session
