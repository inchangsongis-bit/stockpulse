# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

StockPulse is a prototype stock analysis app: a FastAPI backend runs a 4-step AI agent pipeline (news → technicals → sentiment → strategy) to produce BUY/SELL/HOLD signals, and a Next.js frontend renders a multi-ticker dashboard with a price chart, signal history, and news.

Three capabilities have since been added on top of that pipeline, each with its own section below: a **5-minute price-direction forecast** from a locally-trained scikit-learn model, a **daily digest email** to subscribers driven by a scheduled job, and a **free local sentiment path** (FinBERT) that avoids per-article Claude cost.

## Commands

### Backend (`backend/`)

```bash
# Setup — MUST use Python 3.9, not whatever `python3` resolves to.
# pydantic-core==2.9.0 has no prebuilt wheel for Python 3.13+/3.14, and
# building from source fails. Check with `python3 --version` first; on
# this machine /usr/bin/python3 is 3.9, /opt/homebrew/bin/python3 is not.
/usr/bin/python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Run the API (port 8000, seeds mock SPY data into SQLite on first run)
python main.py

# Full test suite
python -m pytest

# Single test file / single test
python -m pytest tests/test_sentiment_agent.py
python -m pytest tests/test_sentiment_agent.py::test_cached_sentiment_only_scores_uncached_articles -v
```

### Frontend (`frontend/`)

```bash
npm install
npm run dev      # port 3000
npm run build    # production build — also does the TS type-check
npm test         # jest + React Testing Library
```

### Both at once

`./run.sh both` (or `backend`/`frontend` alone) handles venv/npm setup and starts both — see the script for exact behavior.

### `.env`

Copied from `.env.example` at the repo root (not `backend/.env` — `config.py` reads `../.env` relative to `backend/`). `ANTHROPIC_API_KEY`, `POLYGON_API_KEY`, `FINNHUB_API_KEY` are all individually optional — see "Mock vs. live mode" below.

## Architecture

### Agent pipeline (`backend/agents/`, orchestrated by `orchestrator.py`)

`POST /api/pipeline/run/{ticker}` (`routers/pipeline.py`) runs four agents in sequence, each a `BaseAgent` subclass with a `run(ticker, context) -> dict` method:

1. **`news_researcher.py`** — fetches articles (mock or live Finnhub)
2. **`quant_analyst.py`** — computes technical indicators from OHLCV (`analysis/indicators.py`, pure pandas/numpy) — independent of step 1, runs off DB data
3. **`sentiment_analyst.py`** — scores each article via Claude (or a keyword-based fallback), consumes step 1's output
4. **`strategy_engine.py`** — combines steps 2 and 3 into a weighted BUY/SELL/HOLD signal with entry/target/stop levels, using Claude for the natural-language reasoning (or a template fallback)

`agents/base.py` has `extract_claude_text()`, used by both Claude-calling agents: it skips non-text content blocks (extended-thinking models put a `thinking` block ahead of the real answer) and strips a ` ```json ` fence the model may wrap the reply in despite being told not to. Always route Claude responses through this rather than `response.content[0].text` directly.

### Mock vs. live mode

Each external dependency degrades independently based on whether its key is configured in `.env`, not a single global flag:

- **News**: `routers/pipeline.py` sets `use_mock = not bool(settings.finnhub_api_key)` and passes it into the pipeline. `news_researcher.py` honors this via `context["use_mock"]`.
- **Sentiment scoring / reasoning**: `sentiment_analyst.py` and `strategy_engine.py` each independently check `settings.anthropic_api_key` themselves (not caller-supplied) and also catch `anthropic.APIError` at the call site, falling back to rule-based scoring / templated reasoning rather than crashing the pipeline on a bad key or rate limit.

**Tests must not hit real APIs.** `.env` in local dev has real keys, and `get_settings()` is `@lru_cache`d, so any test that doesn't monkeypatch `get_settings` (or the relevant client) will attempt a live network call. The established pattern (see `tests/test_sentiment_agent.py`, `tests/test_pipeline_api.py`) is `monkeypatch.setattr("agents.sentiment_analyst.get_settings", lambda: FakeSettings(...))` per module that reads it — `routers/pipeline.py`, `agents/sentiment_analyst.py`, and `agents/strategy_engine.py` each import `get_settings` themselves, so each needs its own patch target.

### OHLCV: daily vs. minute bars share one table

`models.OHLCV` has an `interval` column (`"daily"` | `"minute"`). The indicator pipeline assumes one bar per day, so `routers/pipeline.py`'s OHLCV query is hard-scoped to `interval == "daily"` — minute bars existing for a ticker must never leak into it. `routers/stocks.py`'s `/ohlcv` and `/sync` endpoints both take an `interval` query param; `/sync` deletes-and-replaces only that ticker+interval's rows (so syncing minute data can't wipe daily history or vice versa).

### Real data sources (`backend/data_sources/`)

`polygon.py` (OHLCV) and `finnhub.py` (news) follow the same shape: a module-level `XError` exception, one `async def fetch_...()` that reads its key from `get_settings()`, raises on missing key or non-2xx, and returns plain dicts shaped to unpack directly into the ORM model (`OHLCV(**bar)`). `finnhub.py` also runs article text through `ftfy` (`uncurl_quotes=False`) — some upstream sources Finnhub aggregates (e.g. SeekingAlpha) send double-encoded mojibake text, and this repairs it without flattening legitimately-curly quotes in already-correct text.

### Persistence: only `NewsArticle` and `Signal` are actually used

`models.py` also defines `TechnicalProfile` and `SentimentProfile` — these are **intentionally unused, dead schema** (a deliberate decision, not an oversight: technical indicators are cheap to recompute from OHLCV, and composite sentiment is cheap to recompute from cached per-article scores, so persisting snapshots would have no caching benefit). Don't wire code up to them without confirming that's actually wanted.

`NewsArticle` is the one piece of "expensive" data that's cached and reused, gated to **live pipeline runs only** (`if not use_mock:` in `routers/pipeline.py` — mock runs stay fully ephemeral, no fake articles persisted):
- After a live run, fetched articles + their computed sentiment are upserted keyed by `(ticker, url)`.
- Before scoring, the router pre-loads already-scored articles (by URL) into a `cached_sentiment` dict passed through `orchestrator.run_pipeline()` into `sentiment_analyst.run()`'s context. The sentiment agent skips Claude entirely for cache hits, scoring only genuinely new articles — this is a real cost/latency win on repeat runs, not just an archive.
- `relevance` is always recomputed fresh even for cache hits (it's a function of the *current* fetch's recency window, not a static property of the article) — only `sentiment`/`source_credibility`/`expected_impact`/`reasoning` come from cache.

No Alembic — `database.py`'s `init_db()` runs `Base.metadata.create_all` then a hand-rolled `_add_missing_columns()` that inspects each table and does targeted `ALTER TABLE ... ADD COLUMN` for anything the model has that the on-disk table doesn't. Extend that function's `migrations` list when adding a column to an existing table; `conftest.py`'s test fixtures build tables fresh via `create_all` so tests never need this.

### `services/` — logic shared between HTTP handlers and background jobs

Routers stay thin; anything a scheduled job also needs lives in `services/`
so it can be called directly instead of over HTTP:

- `pipeline_runner.py` — one ticker through the full pipeline. Both
  `POST /api/pipeline/run/{ticker}` and the bulk runner call it. **Tests
  monkeypatching `get_settings` for the pipeline must target
  `services.pipeline_runner.get_settings`, not `routers.pipeline`.**
- `bulk_pipeline.py` — every watchlist ticker in sequence, with in-memory
  progress at `GET /api/pipeline/run-all/status`. A module-level guard
  stops a manual run and the scheduled run from overlapping.
- `ohlcv_sync.py` / `watchlist_summary.py` — extracted from
  `routers/stocks.py` and `routers/watchlist.py` for the same reason.
- `daily_digest.py`, `email_sender.py`, `email_templates.py` — the digest.

### 5-minute forecast (`analysis/forecast*.py`, `routers/forecast.py`)

Separate from the BUY/SELL/HOLD signal and a different question entirely
(5 minutes vs. 2-4 weeks). A `HistGradientBoostingClassifier` trained by
`scripts/train_forecast_model.py` on pooled minute bars; the artifact
lives in `backend/ml/` and is **gitignored** — regenerate it locally or
`GET /api/forecast/{ticker}` returns 503.

Read the conviction logic in `analysis/forecast.py` before changing it.
Measured accuracy is ~51-52% overall, and the apparent edge at high
confidence lives almost entirely in extended-hours illiquidity (61.6%
all-hours vs 52.9% regular-hours at the top 0.1%). Conviction is
therefore **floored to "low" outside regular hours** and capped in the
opening/closing half-hours. Those guards exist because the numbers say
so — don't relax them without re-running `scripts/research_*.py`.

### Daily digest + subscribers

`main.py`'s lifespan registers one APScheduler job at **06:25 US/Pacific**
(DST-aware, 5 minutes before the open) running
`services/daily_digest.py`: incremental OHLCV sync → pipeline for every
watchlist ticker → email each active subscriber. Sending goes through
Resend (`RESEND_API_KEY`); without a verified domain its sandbox sender
only delivers to the Resend account's own address. Unsubscribe is a
token in `Subscriber.unsubscribe_token` — there is no auth system, and
none of the write endpoints are authenticated.

### Sentiment: two paths

`agents/sentiment_analyst.py` scores via Claude (keyword fallback on
error). `analysis/finbert_sentiment.py` is a **free local alternative**
(ProsusAI/finbert, lazy-loaded, ~400MB on first use) behind
`POST /api/stocks/{ticker}/news/score`, which only fills rows where
`sentiment IS NULL` so it never downgrades a Claude-scored article.
Shared heuristics live in `analysis/news_heuristics.py`.

### Query cost on a large `ohlcv` table

The table holds ~13M minute bars after `scripts/backfill_minute_history.py`,
and the graph shows 15 distinct areas read it. Two patterns that were fine
at ~600k rows are not fine now, and both have bitten already:

- **Never fetch-then-filter in Python.** "Latest bar per ticker" by
  streaming every row back took 34s. Use a per-ticker `ORDER BY timestamp
  DESC LIMIT 1` — it hits `ix_ohlcv_ticker_timestamp` and costs ~3ms. A
  `ROW_NUMBER()` window function does *not* use that index.
- **Always bound the row count, not just the date window.** `/ohlcv`
  with `days=1100&interval=minute` returned 428k rows / 52MB before
  `MAX_OHLCV_ROWS` capped it.

Indexes need the same forward-migration treatment as columns —
`create_all` won't add one to an existing table. See
`database.py`'s `_add_missing_indexes`.

### Frontend (`frontend/src/app/page.tsx`)

The entire UI is one file, by convention — `Dashboard` plus its child components (`WatchlistBar`, `PriceChartPanel`, `NewsPanel`, `NewsHistoryPanel`, `SignalCard`, `FactorBreakdown`, `TechnicalPanel`, shared `SentimentBadge`). All components take `ticker` as a prop and fetch their own data in a `useEffect` keyed on it, so adding a new ticker-scoped panel or switching the active ticker doesn't require touching unrelated components.

- API calls go straight to the backend (the `API` const, `NEXT_PUBLIC_API_URL` env var, defaults to `http://localhost:8000`) rather than through Next's own API routes — `next.config.js`'s `rewrites()` config points at the same URL but is currently unused dead config, not the actual request path.
- Selected ticker persists to `localStorage`, restored in a client-only `useEffect` (reading `localStorage` during the initial render would mismatch the server-rendered default and trigger a hydration error).
- `PriceChartPanel` uses `lightweight-charts`, which draws to a real `<canvas>` 2D context that jsdom doesn't implement — `page.test.tsx` mocks the whole module (`jest.mock("lightweight-charts", ...)`) rather than trying to render it, and `jest.setup.js` polyfills `window.matchMedia`/`ResizeObserver`, which the library also needs to even construct a chart instance.
- The test file's `afterEach` does both `jest.clearAllMocks()` (not `resetAllMocks` — that would also wipe the `lightweight-charts` mock's implementation) and `window.localStorage.clear()` (jsdom doesn't reset it between tests, so a ticker switch in one test leaks into the next).
- `mockFetchSequence` (test helper) matches mocked responses by URL substring in insertion order — when one URL is a substring of another (e.g. `/ohlcv` vs. `/api/stocks/AAPL/ohlcv`), list the more specific key first.
