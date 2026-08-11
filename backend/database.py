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
    if "ohlcv" not in inspector.get_table_names():
        return
    existing = {c["name"] for c in inspector.get_columns("ohlcv")}
    if "interval" not in existing:
        sync_conn.execute(text("ALTER TABLE ohlcv ADD COLUMN interval VARCHAR(10) NOT NULL DEFAULT 'daily'"))


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)


async def get_db():
    async with async_session() as session:
        yield session
