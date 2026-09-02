"""
Options data pilot — collect near-the-money SPY contracts so we can test
whether options-derived features (put/call volume ratio, implied
volatility, skew) create an edge during REGULAR trading hours, which is
the only place an edge would be executable.

Why this shape:

  * Polygon exposes no bulk/grouped endpoint for options (it returns 400;
    the equivalent stocks endpoint returns 12,424 tickers in one call), so
    contracts must be fetched one at a time at 5 requests/minute. SPY has
    5,000+ contracts live on any given day, making a full chain
    impossible — one day would take ~17 hours.
  * Narrowing to near-the-money contracts and requesting each contract's
    whole life in a single date-range call brings a 6-month pilot down to
    roughly 1,100 requests, about 4 hours.
  * Open interest and greeks are 403 on this tier, so proper gamma
    exposure is off the table. Implied volatility we can recover
    ourselves by inverting Black-Scholes from the option price, strike,
    expiry and underlying — all of which we do have.

Writes to its own SQLite file (options_pilot.db) rather than the
production database: this is research data that should be trivial to
discard if the pilot shows nothing.

Resumable — contracts already stored are skipped, so an interrupted run
can simply be restarted.

    cd backend && source venv/bin/activate && python scripts/fetch_options_pilot.py
    # optional: --months 3  --strikes 6
"""

import argparse
import asyncio
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
MAIN_DB = BASE / "stockpulse.db"
PILOT_DB = BASE / "options_pilot.db"
UNDERLYING = "SPY"
SECONDS_BETWEEN_REQUESTS = 12.5
# How many days before expiry to start pulling a contract's bars. Weekly
# contracts do most of their volume in the final two weeks.
LOOKBACK_DAYS = 21


def init_db():
    conn = sqlite3.connect(PILOT_DB)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS option_bars (
            contract TEXT NOT NULL, underlying TEXT NOT NULL,
            expiration TEXT NOT NULL, strike REAL NOT NULL, opt_type TEXT NOT NULL,
            timestamp TEXT NOT NULL, open REAL, high REAL, low REAL, close REAL,
            volume INTEGER
        );
        CREATE INDEX IF NOT EXISTS ix_option_bars_ts ON option_bars(timestamp);
        CREATE INDEX IF NOT EXISTS ix_option_bars_contract ON option_bars(contract);
        CREATE TABLE IF NOT EXISTS fetched_contracts (
            contract TEXT PRIMARY KEY, n_bars INTEGER, fetched_at TEXT
        );
    """)
    conn.commit()
    return conn


def spy_close_near(date_str: str):
    """SPY's closing price shortly before an expiry, used to centre the
    strike window. Read from the main DB, which already holds 2 years of
    SPY bars."""
    with sqlite3.connect(f"file:{MAIN_DB}?mode=ro", uri=True) as c:
        row = c.execute(
            "SELECT close FROM ohlcv WHERE ticker='SPY' AND interval='minute' "
            "AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
            (date_str,),
        ).fetchone()
    return row[0] if row else None


async def polygon_get(client, url, params, retries=3):
    for attempt in range(retries):
        r = await client.get(url, params=params)
        if r.status_code == 429:
            await asyncio.sleep(20)
            continue
        if r.status_code != 200:
            return None
        return r.json()
    return None


async def main(months: int, strikes_each_side: int):
    settings = get_settings()
    if not settings.polygon_api_key:
        print("POLYGON_API_KEY is not configured.")
        return
    key = settings.polygon_api_key

    conn = init_db()
    done = {r[0] for r in conn.execute("SELECT contract FROM fetched_contracts")}
    print(f"{len(done)} contracts already stored — those will be skipped\n")

    end = datetime.now().date()
    start = end - timedelta(days=months * 30)

    async with httpx.AsyncClient(timeout=60) as client:
        # Weekly Friday expiries across the window. SPY also lists Mon/Wed
        # expiries, but Fridays carry the most volume and keep the request
        # budget realistic.
        expiries = []
        d = start
        while d <= end:
            if d.weekday() == 4:
                expiries.append(d.isoformat())
            d += timedelta(days=1)
        print(f"{len(expiries)} Friday expiries between {start} and {end}")

        planned = []
        for exp in expiries:
            centre_day = (datetime.fromisoformat(exp) - timedelta(days=10)).date().isoformat()
            spot = spy_close_near(centre_day + " 23:59:59")
            if spot is None:
                print(f"  {exp}: no SPY price near {centre_day} — skipped")
                continue
            lo, hi = spot - strikes_each_side, spot + strikes_each_side
            body = await polygon_get(client, "https://api.polygon.io/v3/reference/options/contracts", {
                "underlying_ticker": UNDERLYING, "expired": "true",
                "expiration_date": exp,
                "strike_price.gte": round(lo), "strike_price.lte": round(hi),
                "limit": 250, "apiKey": key,
            })
            await asyncio.sleep(SECONDS_BETWEEN_REQUESTS)
            results = (body or {}).get("results", [])
            planned.extend(results)
            print(f"  {exp}: spot~{spot:.2f}  strikes {round(lo)}-{round(hi)}  "
                  f"{len(results)} contracts")

        todo = [c for c in planned if c["ticker"] not in done]
        print(f"\n{len(planned)} contracts found, {len(todo)} still to fetch")
        print(f"~{len(todo) * SECONDS_BETWEEN_REQUESTS / 3600:.1f}h at the free-tier rate\n")

        stored = 0
        for i, c in enumerate(todo, 1):
            sym = c["ticker"]
            exp = c["expiration_date"]
            frm = (datetime.fromisoformat(exp) - timedelta(days=LOOKBACK_DAYS)).date().isoformat()
            body = await polygon_get(
                client, f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/minute/{frm}/{exp}",
                {"limit": 50000, "adjusted": "true", "sort": "asc", "apiKey": key},
            )
            bars = (body or {}).get("results") or []
            if bars:
                conn.executemany(
                    "INSERT INTO option_bars (contract, underlying, expiration, strike, "
                    "opt_type, timestamp, open, high, low, close, volume) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    [(sym, UNDERLYING, exp, c["strike_price"], c["contract_type"],
                      datetime.fromtimestamp(b["t"] / 1000).isoformat(),
                      b.get("o"), b.get("h"), b.get("l"), b.get("c"), int(b.get("v", 0)))
                     for b in bars],
                )
            conn.execute(
                "INSERT OR REPLACE INTO fetched_contracts VALUES (?,?,?)",
                (sym, len(bars), datetime.now().isoformat()),
            )
            conn.commit()
            stored += len(bars)
            if i % 10 == 0 or len(bars) > 0:
                print(f"  [{i}/{len(todo)}] {sym}  +{len(bars):>5d} bars  (total {stored:,})")
            await asyncio.sleep(SECONDS_BETWEEN_REQUESTS)

    total = conn.execute("SELECT COUNT(*) FROM option_bars").fetchone()[0]
    contracts = conn.execute("SELECT COUNT(*) FROM fetched_contracts").fetchone()[0]
    print(f"\nDONE. {contracts} contracts, {total:,} option minute bars in {PILOT_DB.name}")
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--strikes", type=int, default=10,
                    help="dollars above/below spot to include (SPY strikes are $1 apart near the money)")
    args = ap.parse_args()
    asyncio.run(main(args.months, args.strikes))
