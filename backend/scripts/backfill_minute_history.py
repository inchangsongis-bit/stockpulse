"""
One-off backfill: pull ~2 years of minute bars for every watchlist ticker,
far beyond what the app's own /sync endpoint allows (it caps minute
requests at 30 days deliberately, since that's all the live product
needs).

The forecast model's learning curve is still climbing at 100% of the 29
days we currently hold, so more history is the cheapest remaining
accuracy improvement available — see scripts/research_forecast_v3.py.

Constraints this works around:
  * Polygon returns at most 50,000 bars per request. Active names produce
    ~600 bars/day including extended hours, so a 60-day chunk (~36k bars)
    stays comfortably inside that with margin.
  * The free tier allows 5 requests/minute, counting every page. This
    paces itself at 12s between requests accordingly.

Idempotent and resumable: before fetching, it checks how many bars are
already stored for each ticker/chunk and skips chunks that already look
complete, so an interrupted run can simply be restarted. Chunks it does
fetch are delete-then-insert over exactly that date range, so re-running
never duplicates rows.

    cd backend && source venv/bin/activate && python scripts/backfill_minute_history.py
    # optional: --years 1  --tickers AAPL,MSFT
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from sqlalchemy import delete, func, select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from database import async_session  # noqa: E402
from models import OHLCV, Watchlist  # noqa: E402

CHUNK_DAYS = 60
SECONDS_BETWEEN_REQUESTS = 12.5  # 5 req/min with a little headroom
# A chunk with at least this many stored bars is treated as already done.
# 60 calendar days is ~40 trading days; even a quiet name clears this.
COMPLETE_ENOUGH = 3000


async def fetch_chunk(client: httpx.AsyncClient, ticker: str, start: str, end: str, api_key: str):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}"
    resp = await client.get(
        url, params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
    )
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"
    body = resp.json()
    if body.get("status") not in ("OK", "DELAYED"):
        return None, f"api status {body.get('status')}"
    results = body.get("results", [])
    if len(results) >= 50000:
        # Would mean the chunk was truncated and we're silently losing
        # bars — surface it rather than storing a partial window.
        return None, "hit the 50k cap (shrink CHUNK_DAYS)"
    return results, None


async def main(years: int, only_tickers):
    settings = get_settings()
    if not settings.polygon_api_key:
        print("POLYGON_API_KEY is not configured.")
        return

    async with async_session() as db:
        result = await db.execute(select(Watchlist.ticker).order_by(Watchlist.ticker))
        tickers = [r[0] for r in result.all()]
    if only_tickers:
        tickers = [t for t in tickers if t in only_tickers]

    today = datetime.now().date()
    # Oldest first, so an interrupted run still leaves a contiguous block.
    chunks = []
    cursor = today - timedelta(days=years * 365)
    while cursor < today:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), today)
        chunks.append((cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + timedelta(days=1)

    total_jobs = len(tickers) * len(chunks)
    print(f"{len(tickers)} tickers x {len(chunks)} chunks of {CHUNK_DAYS}d = {total_jobs} potential requests")
    print(f"Paced at {SECONDS_BETWEEN_REQUESTS}s/request -> up to "
          f"~{total_jobs * SECONDS_BETWEEN_REQUESTS / 3600:.1f}h if nothing is skipped\n")

    fetched = skipped = failed = inserted_total = 0

    async with httpx.AsyncClient(timeout=60) as client:
        for ticker in tickers:
            for start, end in chunks:
                start_dt = datetime.fromisoformat(start)
                end_dt = datetime.fromisoformat(end) + timedelta(days=1)

                async with async_session() as db:
                    existing = await db.execute(
                        select(func.count()).select_from(OHLCV).where(
                            OHLCV.ticker == ticker,
                            OHLCV.interval == "minute",
                            OHLCV.timestamp >= start_dt,
                            OHLCV.timestamp < end_dt,
                        )
                    )
                    have = existing.scalar() or 0

                if have >= COMPLETE_ENOUGH:
                    skipped += 1
                    continue

                results, err = await fetch_chunk(client, ticker, start, end, settings.polygon_api_key)
                fetched += 1

                if err:
                    failed += 1
                    print(f"  {ticker} {start}..{end}  FAILED: {err}")
                else:
                    async with async_session() as db:
                        await db.execute(
                            delete(OHLCV).where(
                                OHLCV.ticker == ticker,
                                OHLCV.interval == "minute",
                                OHLCV.timestamp >= start_dt,
                                OHLCV.timestamp < end_dt,
                            )
                        )
                        for r in results:
                            db.add(OHLCV(
                                ticker=ticker,
                                interval="minute",
                                timestamp=datetime.fromtimestamp(r["t"] / 1000),
                                open=r["o"], high=r["h"], low=r["l"], close=r["c"],
                                volume=int(r["v"]), vwap=r.get("vw"), num_trades=r.get("n"),
                            ))
                        await db.commit()
                    inserted_total += len(results)
                    print(f"  {ticker} {start}..{end}  +{len(results):>6d} bars   "
                          f"[{fetched} fetched / {skipped} skipped / {failed} failed]")

                await asyncio.sleep(SECONDS_BETWEEN_REQUESTS)

    print(f"\nDONE. fetched={fetched} skipped={skipped} failed={failed} "
          f"rows_inserted={inserted_total}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--tickers", type=str, default="")
    args = ap.parse_args()
    only = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
    asyncio.run(main(args.years, only))
