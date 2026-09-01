"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  createChart,
  CrosshairMode,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from "lightweight-charts";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const TICKER_STORAGE_KEY = "stockpulse:ticker";

// ── Types ──

interface OHLCVBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  vwap?: number;
}

interface WatchlistSummaryEntry {
  ticker: string;
  price: number | null;
  signal: { action: string; confidence: number; timestamp: string } | null;
}

interface RunAllStatus {
  running: boolean;
  trigger: string | null;
  total: number;
  completed: number;
  current_ticker: string | null;
  errors: Record<string, string>;
}

interface Signal {
  id: number;
  timestamp: string;
  action: string;
  confidence: number;
  reasoning: string;
  entry_low: number;
  entry_high: number;
  target: number;
  stop_loss: number;
  time_horizon: string;
  risk_level: string;
}

interface PipelineResult {
  ticker: string;
  elapsed_seconds: number;
  signal: {
    action: string;
    confidence: number;
    reasoning: string;
    entry_low: number;
    entry_high: number;
    target: number;
    stop_loss: number;
    time_horizon: string;
    risk_level: string;
    composite_score: number;
    factors: Record<string, { score: number; weight: number; weighted: number }>;
    disclaimer: string;
  };
  technical_profile: {
    trend_score: number;
    momentum_score: number;
    volatility_state: string;
    volume_anomaly: boolean;
    patterns: string[];
    indicators: Record<string, number>;
  };
  sentiment_profile: {
    composite_sentiment: number;
    sentiment_trend: string;
    article_scores: Array<{
      title: string;
      source: string;
      url: string;
      sentiment: number;
      expected_impact: string;
    }>;
  };
  research: {
    articles: Array<{
      title: string;
      source: string;
      url: string;
      summary: string;
      category: string;
      relevance: number;
    }>;
  };
}

// ── Main Dashboard ──

export default function Dashboard() {
  const [ticker, setTicker] = useState("SPY");
  const [summary, setSummary] = useState<WatchlistSummaryEntry[]>([]);
  const [watchlistError, setWatchlistError] = useState<string | null>(null);
  const [addingTicker, setAddingTicker] = useState(false);
  const [ohlcv, setOhlcv] = useState<OHLCVBar[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [pipelineResult, setPipelineResult] = useState<PipelineResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"signal" | "technical" | "news">("signal");
  const [refreshAllStatus, setRefreshAllStatus] = useState<RunAllStatus | null>(null);

  // Watchlist tickers + their latest price/signal, for the BUY/SELL/HOLD
  // grouped overview — one call instead of 2 requests per ticker.
  const fetchSummary = useCallback(() => {
    return fetch(`${API}/api/watchlist/summary`)
      .then((r) => r.json())
      .then((d) => setSummary(d.tickers || []))
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetchSummary();
  }, [fetchSummary]);

  const refreshAll = useCallback(async () => {
    try {
      await fetch(`${API}/api/pipeline/run-all`, { method: "POST" });
    } catch {
      // The status poll below will simply keep showing "idle" — acceptable
      // no-op if the backend is unreachable.
    }
  }, []);

  // Poll run-all progress continuously (cheap local GET) so both a manual
  // "Refresh All" click and the daily scheduled job (which runs with no
  // user action) show up here the same way.
  useEffect(() => {
    let cancelled = false;
    let wasRunning = false;

    const poll = async () => {
      try {
        const res = await fetch(`${API}/api/pipeline/run-all/status`);
        const data: RunAllStatus = await res.json();
        if (cancelled) return;
        setRefreshAllStatus(data);
        if (data.running) {
          wasRunning = true;
        } else if (wasRunning) {
          wasRunning = false;
          fetchSummary(); // pick up newly-generated signals
        }
      } catch {
        // ignore transient poll failures
      }
    };

    poll();
    const id = setInterval(poll, 3000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [fetchSummary]);

  // Restore the last-selected ticker after mount (client-only — reading
  // localStorage during the initial render would mismatch the server-
  // rendered "SPY" default and trigger a hydration error).
  useEffect(() => {
    const saved = window.localStorage.getItem(TICKER_STORAGE_KEY);
    if (saved) setTicker(saved);
  }, []);

  useEffect(() => {
    window.localStorage.setItem(TICKER_STORAGE_KEY, ticker);
  }, [ticker]);

  // A pipeline result / error belongs to whichever ticker produced it —
  // don't leave it showing after switching to a different ticker.
  useEffect(() => {
    setPipelineResult(null);
    setError(null);
  }, [ticker]);

  const addTicker = useCallback(async (newTicker: string) => {
    setWatchlistError(null);
    setAddingTicker(true);
    try {
      const res = await fetch(`${API}/api/watchlist/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: newTicker }),
      });
      const data = await res.json();
      if (!res.ok) {
        setWatchlistError(data.detail || `Could not add ${newTicker}.`);
        return;
      }
      setSummary((s) => [...s, { ticker: data.ticker, price: null, signal: null }]);

      // Best-effort: pull real daily data so the ticker isn't empty by
      // default. If this fails (e.g. an invalid symbol upstream), the
      // ticker is still added — the chart's own empty state offers a
      // manual "Sync Live Data" retry.
      try {
        await fetch(`${API}/api/stocks/${data.ticker}/sync?interval=daily`, { method: "POST" });
      } catch {
        // ignored — see comment above
      }

      setTicker(data.ticker);
    } catch {
      setWatchlistError("Failed to add ticker — is the backend running?");
    } finally {
      setAddingTicker(false);
    }
  }, []);

  const removeTicker = useCallback(
    async (target: string) => {
      setWatchlistError(null);
      try {
        const res = await fetch(`${API}/api/watchlist/${target}`, { method: "DELETE" });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          setWatchlistError(data.detail || `Could not remove ${target}.`);
          return;
        }
        const next = summary.filter((entry) => entry.ticker !== target);
        setSummary(next);
        if (target === ticker && next.length > 0) {
          setTicker(next[0].ticker);
        }
      } catch {
        setWatchlistError("Failed to remove ticker — is the backend running?");
      }
    },
    [ticker, summary]
  );

  // Fetch OHLCV data
  useEffect(() => {
    fetch(`${API}/api/stocks/${ticker}/ohlcv?days=365`)
      .then((r) => r.json())
      .then((d) => setOhlcv(d.data || []))
      .catch((e) => setError("Backend not running. Start with: cd backend && python main.py"));
  }, [ticker]);

  // Fetch existing signals
  useEffect(() => {
    fetch(`${API}/api/signals/${ticker}`)
      .then((r) => r.json())
      .then((d) => setSignals(d.signals || []))
      .catch(() => {});
  }, [ticker]);

  // Run pipeline
  const runPipeline = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await fetch(`${API}/api/pipeline/run/${ticker}`, { method: "POST" });
      if (!res.ok) {
        setError(`Pipeline failed (${res.status}). Check the backend logs for details.`);
        return;
      }
      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else {
        setPipelineResult(data);
        // Refresh signals
        const sigRes = await fetch(`${API}/api/signals/${ticker}`);
        const sigData = await sigRes.json();
        setSignals(sigData.signals || []);
      }
    } catch (e) {
      setError("Failed to run pipeline. Is the backend running?");
    } finally {
      setRunning(false);
    }
  }, [ticker]);

  const latestPrice = ohlcv.length > 0 ? ohlcv[ohlcv.length - 1].close : 0;
  const prevPrice = ohlcv.length > 1 ? ohlcv[ohlcv.length - 2].close : latestPrice;
  const priceChange = latestPrice - prevPrice;
  const pctChange = prevPrice ? (priceChange / prevPrice) * 100 : 0;

  return (
    <div className="min-h-screen p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-3">
            StockPulse
            <span className="text-sm font-normal px-2 py-1 rounded bg-[var(--bg-card)] border border-[var(--border)] text-[var(--text-muted)]">
              v0.1 Prototype
            </span>
          </h1>
          <p className="text-sm text-[var(--text-muted)] mt-1">
            AI-powered stock analysis — for informational purposes only
          </p>
        </div>
        <button
          onClick={runPipeline}
          disabled={running}
          className={`px-5 py-2.5 rounded-lg font-medium text-sm transition-all ${
            running
              ? "bg-[var(--bg-hover)] text-[var(--text-muted)] cursor-wait"
              : "bg-[var(--blue)] text-white hover:brightness-110"
          }`}
        >
          {running ? "Running Pipeline..." : "Run Analysis Pipeline"}
        </button>
      </div>

      <TickerSearchBox
        tickers={summary.map((s) => s.ticker)}
        onSelect={setTicker}
        onAdd={addTicker}
        adding={addingTicker}
        error={watchlistError}
      />

      <SignalOverviewPanel
        summary={summary}
        activeTicker={ticker}
        onSelect={setTicker}
        onRemove={removeTicker}
        onRefreshAll={refreshAll}
        refreshStatus={refreshAllStatus}
      />

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Ticker Header */}
      <div className="flex items-end gap-4 mb-6">
        <span className="text-3xl font-bold">{ticker}</span>
        <span className="text-2xl font-mono">${latestPrice.toFixed(2)}</span>
        <span className={`text-lg font-mono ${priceChange >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"}`}>
          {priceChange >= 0 ? "+" : ""}
          {priceChange.toFixed(2)} ({pctChange.toFixed(2)}%)
        </span>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Chart — spans 2 cols */}
        <div className="lg:col-span-2">
          <PriceChartPanel ticker={ticker} />
        </div>

        {/* Signal Card */}
        <div>
          {pipelineResult ? (
            <SignalCard signal={pipelineResult.signal} />
          ) : signals.length > 0 ? (
            <SignalCard
              signal={{
                action: signals[0].action,
                confidence: signals[0].confidence,
                reasoning: signals[0].reasoning,
                entry_low: signals[0].entry_low,
                entry_high: signals[0].entry_high,
                target: signals[0].target,
                stop_loss: signals[0].stop_loss,
                time_horizon: signals[0].time_horizon,
                risk_level: signals[0].risk_level,
                composite_score: 0,
                factors: {},
                disclaimer: "For informational purposes only.",
              }}
            />
          ) : (
            <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-6 text-center text-[var(--text-muted)]">
              <p className="text-lg mb-2">No signals yet</p>
              <p className="text-sm">Click "Run Analysis Pipeline" to generate</p>
            </div>
          )}
        </div>
      </div>

      {/* Tabs: Technical / Sentiment / News */}
      {pipelineResult && (
        <div className="mt-6">
          <div className="flex gap-1 mb-4 border-b border-[var(--border)]">
            {(["signal", "technical", "news"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 text-sm font-medium capitalize transition-colors ${
                  activeTab === tab
                    ? "text-[var(--blue)] border-b-2 border-[var(--blue)]"
                    : "text-[var(--text-muted)] hover:text-[var(--text)]"
                }`}
              >
                {tab === "signal" ? "Factor Breakdown" : tab === "technical" ? "Technical Indicators" : "News & Sentiment"}
              </button>
            ))}
          </div>

          {activeTab === "signal" && <FactorBreakdown factors={pipelineResult.signal.factors} />}
          {activeTab === "technical" && <TechnicalPanel profile={pipelineResult.technical_profile} />}
          {activeTab === "news" && (
            <NewsPanel
              articles={pipelineResult.research.articles}
              sentimentScores={pipelineResult.sentiment_profile.article_scores}
            />
          )}
        </div>
      )}

      {/* Signal History */}
      {signals.length > 0 && (
        <div className="mt-6">
          <h2 className="text-lg font-semibold mb-3">Signal History</h2>
          <div className="space-y-2">
            {signals.map((s) => (
              <div
                key={s.id}
                className="flex items-center gap-4 p-3 rounded-lg bg-[var(--bg-card)] border border-[var(--border)]"
              >
                <ActionBadge action={s.action} />
                <span className="text-sm font-mono text-[var(--text-muted)]">
                  {new Date(s.timestamp).toLocaleString()}
                </span>
                <span className="text-sm">Confidence: {s.confidence}%</span>
                <span className="text-sm text-[var(--text-muted)] flex-1 truncate">
                  {s.reasoning}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <NewsHistoryPanel ticker={ticker} />
    </div>
  );
}

// ── Components ──

// Compact search-or-add combobox — replaces the old row of ticker chips.
// Typing filters the existing watchlist; if nothing matches exactly, an
// "Add" option appears to add it as a new ticker.
function TickerSearchBox({
  tickers,
  onSelect,
  onAdd,
  adding,
  error,
}: {
  tickers: string[];
  onSelect: (ticker: string) => void;
  onAdd: (ticker: string) => void;
  adding: boolean;
  error: string | null;
}) {
  const [input, setInput] = useState("");
  const [open, setOpen] = useState(false);

  const query = input.trim().toUpperCase();
  const matches = query ? tickers.filter((t) => t.includes(query)).slice(0, 8) : [];
  const exactMatch = tickers.includes(query);

  const handleSelect = (t: string) => {
    onSelect(t);
    setInput("");
    setOpen(false);
  };

  const handleAdd = () => {
    if (!query || adding) return;
    onAdd(query);
    setInput("");
    setOpen(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== "Enter") return;
    if (exactMatch) handleSelect(query);
    else handleAdd();
  };

  return (
    <div className="relative mb-4 max-w-xs">
      <input
        value={input}
        onChange={(e) => {
          setInput(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={handleKeyDown}
        placeholder="Search or add ticker"
        disabled={adding}
        className="w-full px-3 py-2 rounded-lg text-sm bg-[var(--bg-card)] border border-[var(--border)] text-[var(--text)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--blue)] disabled:opacity-50"
      />
      {open && query && (matches.length > 0 || !exactMatch) && (
        <div className="absolute z-10 mt-1 w-full rounded-lg border border-[var(--border)] bg-[var(--bg-card)] shadow-lg overflow-hidden">
          {matches.map((t) => (
            <button
              key={t}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => handleSelect(t)}
              className="block w-full text-left px-3 py-2 text-sm text-[var(--text)] hover:bg-[var(--bg-hover)]"
            >
              {t}
            </button>
          ))}
          {!exactMatch && (
            <button
              onMouseDown={(e) => e.preventDefault()}
              onClick={handleAdd}
              disabled={adding}
              className="block w-full text-left px-3 py-2 text-sm text-[var(--blue)] hover:bg-[var(--bg-hover)] border-t border-[var(--border)] disabled:opacity-50"
            >
              {adding ? "Adding…" : `+ Add "${query}" to watchlist`}
            </button>
          )}
        </div>
      )}
      {error && <p className="text-xs text-[var(--red)] mt-1.5">{error}</p>}
    </div>
  );
}

type SignalFilter = "all" | "BUY" | "SELL" | "HOLD" | "none";

// The BUY/SELL/HOLD grouped watchlist overview — segmented filter tabs
// over a compact, scrollable list (see CLAUDE.md-style rationale: this
// scales far better than a card grid or a chip row once the watchlist
// has 50+ tickers).
function SignalOverviewPanel({
  summary,
  activeTicker,
  onSelect,
  onRemove,
  onRefreshAll,
  refreshStatus,
}: {
  summary: WatchlistSummaryEntry[];
  activeTicker: string;
  onSelect: (ticker: string) => void;
  onRemove: (ticker: string) => void;
  onRefreshAll: () => void;
  refreshStatus: RunAllStatus | null;
}) {
  const [filter, setFilter] = useState<SignalFilter>("all");

  const counts = {
    all: summary.length,
    BUY: summary.filter((s) => s.signal?.action === "BUY").length,
    SELL: summary.filter((s) => s.signal?.action === "SELL").length,
    HOLD: summary.filter((s) => s.signal?.action === "HOLD").length,
    none: summary.filter((s) => !s.signal).length,
  };

  const filtered =
    filter === "all"
      ? summary
      : filter === "none"
      ? summary.filter((s) => !s.signal)
      : summary.filter((s) => s.signal?.action === filter);

  const refreshing = refreshStatus?.running ?? false;

  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-4 mb-6">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <h2 className="text-sm font-medium text-[var(--text-muted)]">Watchlist Signals</h2>
        <button
          onClick={onRefreshAll}
          disabled={refreshing}
          className="px-3 py-1 rounded text-xs font-medium bg-[var(--bg-hover)] border border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)] disabled:opacity-50 disabled:cursor-wait"
        >
          {refreshing
            ? `Refreshing… ${refreshStatus!.completed}/${refreshStatus!.total}${
                refreshStatus!.current_ticker ? ` (${refreshStatus!.current_ticker})` : ""
              }`
            : "Refresh All"}
        </button>
      </div>

      <div className="flex flex-wrap gap-1 mb-3">
        {(
          [
            ["all", "All"],
            ["BUY", "BUY"],
            ["SELL", "SELL"],
            ["HOLD", "HOLD"],
            ["none", "No Signal"],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setFilter(key)}
            className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
              filter === key
                ? "bg-[var(--blue)] text-white"
                : "bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text)]"
            }`}
          >
            {label} ({counts[key]})
          </button>
        ))}
      </div>

      <div className="max-h-72 overflow-y-auto rounded-lg border border-[var(--border)] divide-y divide-[var(--border)]">
        {filtered.length === 0 ? (
          <p className="p-4 text-sm text-[var(--text-muted)] text-center">No tickers in this category.</p>
        ) : (
          filtered.map((s) => (
            <div
              key={s.ticker}
              className={`flex items-center gap-3 px-3 py-2 text-sm transition-colors ${
                s.ticker === activeTicker ? "bg-[var(--blue)]/10" : "hover:bg-[var(--bg-hover)]"
              }`}
            >
              <button
                onClick={() => onSelect(s.ticker)}
                className={`font-semibold w-16 text-left ${
                  s.ticker === activeTicker ? "text-[var(--blue)]" : "text-[var(--text)]"
                }`}
              >
                {s.ticker}
              </button>
              <span className="w-20 font-mono text-[var(--text-muted)]">
                {s.price != null ? `$${s.price.toFixed(2)}` : "—"}
              </span>
              <span className="flex-1 flex items-center gap-2">
                {s.signal ? (
                  <>
                    <ActionBadge action={s.signal.action} />
                    <span className="text-xs text-[var(--text-muted)]">{s.signal.confidence}%</span>
                  </>
                ) : (
                  <span className="text-xs text-[var(--text-muted)]">Not analyzed yet</span>
                )}
              </span>
              {summary.length > 1 && (
                <button
                  onClick={() => onRemove(s.ticker)}
                  aria-label={`Remove ${s.ticker}`}
                  className="w-5 h-5 flex items-center justify-center rounded-full text-xs leading-none text-[var(--text-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--text)]"
                >
                  ×
                </button>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

type ChartInterval = "daily" | "minute";
type ChartType = "area" | "candlestick";

interface RangeOption {
  label: string;
  days: number;
}

const DAILY_RANGES: RangeOption[] = [
  { label: "1M", days: 30 },
  { label: "3M", days: 90 },
  { label: "6M", days: 180 },
  { label: "1Y", days: 365 },
  { label: "2Y", days: 730 },
];

const MINUTE_RANGES: RangeOption[] = [
  { label: "1D", days: 1 },
  { label: "3D", days: 3 },
  { label: "5D", days: 5 },
];

function toUnixSeconds(iso: string): UTCTimestamp {
  return Math.floor(new Date(iso).getTime() / 1000) as UTCTimestamp;
}

function PriceChartPanel({ ticker }: { ticker: string }) {
  const [interval, setInterval_] = useState<ChartInterval>("daily");
  const [chartType, setChartType] = useState<ChartType>("area");
  const [days, setDays] = useState(365);
  const [bars, setBars] = useState<OHLCVBar[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [syncMsg, setSyncMsg] = useState<string | null>(null);
  const [hover, setHover] = useState<OHLCVBar | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const areaSeriesRef = useRef<ISeriesApi<"Area"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const barsRef = useRef<OHLCVBar[]>([]);

  const ranges = interval === "daily" ? DAILY_RANGES : MINUTE_RANGES;

  // Reset to a sensible default range when the interval changes
  useEffect(() => {
    setDays(interval === "daily" ? 365 : 3);
  }, [interval]);

  const loadBars = useCallback(() => {
    setLoading(true);
    setLoadError(null);
    fetch(`${API}/api/stocks/${ticker}/ohlcv?days=${days}&interval=${interval}`)
      .then((r) => r.json())
      .then((d) => setBars(d.data || []))
      .catch(() => setLoadError("Failed to load chart data."))
      .finally(() => setLoading(false));
  }, [ticker, days, interval]);

  // A stale sync confirmation from a previous ticker/interval shouldn't
  // linger once the view has moved on — but this must NOT fire from
  // loadBars() itself, since handleSync calls loadBars() right after
  // setting the success message for the *same* ticker/interval.
  useEffect(() => {
    setSyncMsg(null);
  }, [ticker, interval]);

  useEffect(() => {
    loadBars();
  }, [loadBars]);

  // Create the chart once on mount
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      autoSize: true,
      layout: { background: { color: "transparent" }, textColor: "#8888a0", fontSize: 11 },
      grid: {
        vertLines: { color: "#1a1a2e" },
        horzLines: { color: "#1a1a2e" },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#1a1a2e" },
      timeScale: { borderColor: "#1a1a2e" },
      // Robinhood-style: a one-finger drag scrubs the crosshair instead of
      // panning the chart (pinch-to-zoom is unaffected — that's handleScale,
      // not handleScroll).
      handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: false, vertTouchDrag: false },
    });

    // Overlay histogram on its own (invisible) price scale so it shares
    // the chart without competing with the price series' own axis.
    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "",
    });
    volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

    // Series-type-agnostic: look the hovered bar up from our own data by
    // timestamp rather than reading it back out of the series (whose shape
    // differs between candlestick and area), so this doesn't need to know
    // which price series is currently active.
    chart.subscribeCrosshairMove((param) => {
      const latest = barsRef.current;
      if (!param.time) {
        setHover(latest.length > 0 ? latest[latest.length - 1] : null);
        return;
      }
      const match = latest.find((b) => toUnixSeconds(b.timestamp) === param.time);
      setHover(match ?? (latest.length > 0 ? latest[latest.length - 1] : null));
    });

    chartRef.current = chart;
    volumeSeriesRef.current = volumeSeries;

    return () => {
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      areaSeriesRef.current = null;
      volumeSeriesRef.current = null;
    };
  }, []);

  // Create/replace the price series whenever the chart type toggles —
  // candlestick and area series have incompatible data shapes, so this
  // can't just be a restyle of one persistent series.
  useEffect(() => {
    if (!chartRef.current) return;

    let series: ISeriesApi<"Area"> | ISeriesApi<"Candlestick">;
    if (chartType === "area") {
      series = chartRef.current.addAreaSeries({
        lineColor: "#22c55e",
        topColor: "rgba(34,197,94,0.28)",
        bottomColor: "rgba(34,197,94,0)",
        lineWidth: 2,
      });
      areaSeriesRef.current = series;
      candleSeriesRef.current = null;
    } else {
      series = chartRef.current.addCandlestickSeries({
        upColor: "#22c55e",
        downColor: "#ef4444",
        borderUpColor: "#22c55e",
        borderDownColor: "#ef4444",
        wickUpColor: "#22c55e",
        wickDownColor: "#ef4444",
      });
      candleSeriesRef.current = series;
      areaSeriesRef.current = null;
    }

    return () => {
      chartRef.current?.removeSeries(series);
    };
  }, [chartType]);

  // Push fresh data into the chart whenever bars/interval/chartType change
  useEffect(() => {
    barsRef.current = bars;
    if (!volumeSeriesRef.current || !chartRef.current) return;

    volumeSeriesRef.current.setData(
      bars.map((b) => ({
        time: toUnixSeconds(b.timestamp),
        value: b.volume,
        color: b.close >= b.open ? "rgba(34,197,94,0.4)" : "rgba(239,68,68,0.4)",
      }))
    );

    if (chartType === "candlestick" && candleSeriesRef.current) {
      candleSeriesRef.current.setData(
        bars.map((b) => ({
          time: toUnixSeconds(b.timestamp),
          open: b.open,
          high: b.high,
          low: b.low,
          close: b.close,
        }))
      );
    } else if (chartType === "area" && areaSeriesRef.current) {
      // Robinhood colors the whole line/fill by whether price is up or
      // down over the currently-loaded range, not per-bar.
      const up = bars.length === 0 || bars[bars.length - 1].close >= bars[0].open;
      areaSeriesRef.current.applyOptions({
        lineColor: up ? "#22c55e" : "#ef4444",
        topColor: up ? "rgba(34,197,94,0.28)" : "rgba(239,68,68,0.28)",
        bottomColor: up ? "rgba(34,197,94,0)" : "rgba(239,68,68,0)",
      });
      areaSeriesRef.current.setData(
        bars.map((b) => ({ time: toUnixSeconds(b.timestamp), value: b.close }))
      );
    }

    chartRef.current.applyOptions({
      timeScale: { timeVisible: interval === "minute", secondsVisible: false },
    });
    chartRef.current.timeScale().fitContent();

    setHover(bars.length > 0 ? bars[bars.length - 1] : null);
  }, [bars, interval, chartType]);

  const handleSync = async () => {
    setSyncing(true);
    setSyncMsg(null);
    try {
      const res = await fetch(
        `${API}/api/stocks/${ticker}/sync?interval=${interval}&days=${days}`,
        { method: "POST" }
      );
      const data = await res.json();
      if (!res.ok) {
        setSyncMsg(data.detail || `Sync failed (${res.status}).`);
      } else {
        setSyncMsg(`Synced ${data.synced_bars} bars — latest close $${data.latest_close.toFixed(2)}`);
        loadBars();
      }
    } catch {
      setSyncMsg("Sync failed — is the backend running?");
    } finally {
      setSyncing(false);
    }
  };

  // Per-bar change (this bar's own move), used in the detailed OHLC row.
  const change = hover ? hover.close - hover.open : 0;
  const changePct = hover && hover.open ? (change / hover.open) * 100 : 0;

  // Change since the start of the currently-loaded range, used for the
  // big Robinhood-style header price (updates live as the crosshair moves).
  const periodOpen = bars.length > 0 ? bars[0].open : null;
  const periodChange = hover && periodOpen != null ? hover.close - periodOpen : 0;
  const periodChangePct = periodOpen ? (periodChange / periodOpen) * 100 : 0;

  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-4">
      {/* Header row: big price + filters */}
      <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
        <div>
          <h2 className="text-xs font-medium text-[var(--text-muted)] mb-1">{ticker} · Price Chart</h2>
          {hover && (
            <div className="flex items-baseline gap-2 flex-wrap">
              <span className="text-3xl font-semibold text-[var(--text)] tabular-nums">
                ${hover.close.toFixed(2)}
              </span>
              <span
                className={`text-sm font-medium tabular-nums ${
                  periodChange >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"
                }`}
              >
                {periodChange >= 0 ? "+" : ""}
                {periodChange.toFixed(2)} ({periodChangePct.toFixed(2)}%)
              </span>
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Interval toggle */}
          <div className="flex rounded-lg border border-[var(--border)] overflow-hidden">
            {(["daily", "minute"] as const).map((iv) => (
              <button
                key={iv}
                onClick={() => setInterval_(iv)}
                className={`px-3 py-1 text-xs font-medium transition-colors ${
                  interval === iv
                    ? "bg-[var(--blue)] text-white"
                    : "bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text)]"
                }`}
              >
                {iv === "daily" ? "Daily" : "1-Minute"}
              </button>
            ))}
          </div>

          {/* Chart type toggle */}
          <div className="flex rounded-lg border border-[var(--border)] overflow-hidden">
            {(["area", "candlestick"] as const).map((ct) => (
              <button
                key={ct}
                onClick={() => setChartType(ct)}
                className={`px-3 py-1 text-xs font-medium transition-colors ${
                  chartType === ct
                    ? "bg-[var(--blue)] text-white"
                    : "bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text)]"
                }`}
              >
                {ct === "area" ? "Line" : "Candles"}
              </button>
            ))}
          </div>

          {/* Range pills */}
          <div className="flex gap-1">
            {ranges.map((r) => (
              <button
                key={r.label}
                onClick={() => setDays(r.days)}
                className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                  days === r.days
                    ? "bg-[var(--bg-hover)] text-[var(--text)] border border-[var(--border)]"
                    : "text-[var(--text-muted)] hover:text-[var(--text)]"
                }`}
              >
                {r.label}
              </button>
            ))}
          </div>

          {/* Sync button */}
          <button
            onClick={handleSync}
            disabled={syncing}
            className="px-3 py-1 rounded text-xs font-medium bg-[var(--bg-hover)] border border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)] disabled:opacity-50"
          >
            {syncing ? "Syncing…" : "Sync Live Data"}
          </button>
        </div>
      </div>

      {syncMsg && <p className="text-xs text-[var(--text-muted)] mb-2">{syncMsg}</p>}

      {/* OHLC legend / readout — tracks the crosshair, defaults to the latest bar */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mb-2 text-xs font-mono min-h-[16px]">
        {hover && (
          <>
            <span className="text-[var(--text-muted)]">
              {interval === "minute"
                ? new Date(hover.timestamp).toLocaleString()
                : new Date(hover.timestamp).toLocaleDateString()}
            </span>
            <span>
              O <span className="text-[var(--text)]">{hover.open.toFixed(2)}</span>
            </span>
            <span>
              H <span className="text-[var(--text)]">{hover.high.toFixed(2)}</span>
            </span>
            <span>
              L <span className="text-[var(--text)]">{hover.low.toFixed(2)}</span>
            </span>
            <span>
              C{" "}
              <span className={change >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"}>
                {hover.close.toFixed(2)}
              </span>
            </span>
            <span className={change >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"}>
              {change >= 0 ? "+" : ""}
              {change.toFixed(2)} ({changePct.toFixed(2)}%)
            </span>
            <span className="text-[var(--text-muted)]">Vol {hover.volume.toLocaleString()}</span>
          </>
        )}
      </div>

      {/* Chart canvas, with loading/empty overlays */}
      <div className="relative">
        <div ref={containerRef} className="w-full" style={{ height: 380 }} />
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-[var(--bg-card)]/70 text-sm text-[var(--text-muted)]">
            Loading…
          </div>
        )}
        {!loading && bars.length === 0 && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-sm text-[var(--text-muted)] bg-[var(--bg-card)]">
            <p>{loadError || `No ${interval} data yet for ${ticker}.`}</p>
            <button
              onClick={handleSync}
              disabled={syncing}
              className="px-3 py-1.5 rounded-lg text-xs font-medium bg-[var(--blue)] text-white hover:brightness-110 disabled:opacity-50"
            >
              {syncing ? "Syncing…" : "Sync Live Data"}
            </button>
          </div>
        )}
      </div>

      {/* Color legend */}
      <div className="flex items-center gap-4 mt-3 pt-3 border-t border-[var(--border)] text-xs text-[var(--text-muted)]">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: "#22c55e" }} />
          Bullish (close ≥ open)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: "#ef4444" }} />
          Bearish (close &lt; open)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 rounded-sm bg-[var(--text-muted)] opacity-40" />
          Volume
        </span>
      </div>
    </div>
  );
}

function SignalCard({ signal }: { signal: PipelineResult["signal"] }) {
  const actionColors: Record<string, string> = {
    BUY: "var(--green)",
    SELL: "var(--red)",
    HOLD: "var(--yellow)",
  };

  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-medium text-[var(--text-muted)]">Latest Signal</h2>
        <span
          className="text-xs px-2 py-0.5 rounded-full border"
          style={{
            borderColor: signal.risk_level === "high" ? "var(--red)" : signal.risk_level === "moderate" ? "var(--yellow)" : "var(--green)",
            color: signal.risk_level === "high" ? "var(--red)" : signal.risk_level === "moderate" ? "var(--yellow)" : "var(--green)",
          }}
        >
          {signal.risk_level} risk
        </span>
      </div>

      <div className="flex items-center gap-3 mb-4">
        <span
          className="text-3xl font-bold"
          style={{ color: actionColors[signal.action] || "var(--text)" }}
        >
          {signal.action}
        </span>
        <div className="flex flex-col">
          <span className="text-sm text-[var(--text-muted)]">Confidence</span>
          <span className="text-xl font-mono font-semibold">{signal.confidence}%</span>
        </div>
      </div>

      {/* Confidence bar */}
      <div className="w-full h-2 rounded-full bg-[var(--bg-hover)] mb-4">
        <div
          className="h-full rounded-full transition-all"
          style={{
            width: `${signal.confidence}%`,
            backgroundColor: actionColors[signal.action],
          }}
        />
      </div>

      <p className="text-sm text-[var(--text-muted)] mb-4 leading-relaxed">{signal.reasoning}</p>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <span className="text-[var(--text-muted)]">Entry Range</span>
          <p className="font-mono">${signal.entry_low} — ${signal.entry_high}</p>
        </div>
        <div>
          <span className="text-[var(--text-muted)]">Target</span>
          <p className="font-mono text-[var(--green)]">${signal.target}</p>
        </div>
        <div>
          <span className="text-[var(--text-muted)]">Stop Loss</span>
          <p className="font-mono text-[var(--red)]">${signal.stop_loss}</p>
        </div>
        <div>
          <span className="text-[var(--text-muted)]">Horizon</span>
          <p>{signal.time_horizon}</p>
        </div>
      </div>

      <p className="text-xs text-[var(--text-muted)] mt-4 pt-3 border-t border-[var(--border)]">
        {signal.disclaimer}
      </p>
    </div>
  );
}

function FactorBreakdown({ factors }: { factors: Record<string, { score: number; weight: number; weighted: number }> }) {
  if (!factors || Object.keys(factors).length === 0) return null;

  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-5">
      <h3 className="text-sm font-medium text-[var(--text-muted)] mb-4">Weighted Factor Scores</h3>
      <div className="space-y-4">
        {Object.entries(factors).map(([name, f]) => (
          <div key={name}>
            <div className="flex justify-between text-sm mb-1">
              <span className="capitalize">{name}</span>
              <span className="font-mono">
                {f.score.toFixed(3)} x {(f.weight * 100).toFixed(0)}% = {f.weighted.toFixed(3)}
              </span>
            </div>
            <div className="w-full h-3 rounded-full bg-[var(--bg-hover)] relative overflow-hidden">
              <div
                className="absolute top-0 h-full rounded-full transition-all"
                style={{
                  left: f.score >= 0 ? "50%" : `${50 + f.score * 50}%`,
                  width: `${Math.abs(f.score) * 50}%`,
                  backgroundColor: f.score >= 0 ? "var(--green)" : "var(--red)",
                }}
              />
              {/* Center line */}
              <div className="absolute top-0 left-1/2 w-px h-full bg-[var(--text-muted)] opacity-40" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TechnicalPanel({ profile }: { profile: PipelineResult["technical_profile"] }) {
  const ind = profile.indicators || {};
  const items = [
    { label: "RSI (14)", value: ind.rsi_14, warn: ind.rsi_14 > 70 || ind.rsi_14 < 30 },
    { label: "MACD Histogram", value: ind.macd_histogram },
    { label: "SMA 20", value: ind.sma_20 },
    { label: "SMA 50", value: ind.sma_50 },
    { label: "SMA 200", value: ind.sma_200 },
    { label: "ATR (14)", value: ind.atr_14 },
    { label: "ADX (14)", value: ind.adx_14 },
    { label: "Stochastic %K", value: ind.stochastic_k },
    { label: "ROC (12)", value: ind.roc_12 },
    { label: "Vol/SMA Ratio", value: ind.volume_sma_ratio },
    { label: "Bollinger BW", value: ind.bollinger_bandwidth },
  ];

  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-5">
      <div className="flex gap-6 mb-4">
        <div>
          <span className="text-xs text-[var(--text-muted)]">Trend Score</span>
          <p className={`text-xl font-mono font-semibold ${profile.trend_score > 0 ? "text-[var(--green)]" : "text-[var(--red)]"}`}>
            {profile.trend_score.toFixed(3)}
          </p>
        </div>
        <div>
          <span className="text-xs text-[var(--text-muted)]">Momentum</span>
          <p className={`text-xl font-mono font-semibold ${profile.momentum_score > 0 ? "text-[var(--green)]" : "text-[var(--red)]"}`}>
            {profile.momentum_score.toFixed(3)}
          </p>
        </div>
        <div>
          <span className="text-xs text-[var(--text-muted)]">Volatility</span>
          <p className="text-xl font-semibold capitalize">{profile.volatility_state}</p>
        </div>
        {profile.patterns.length > 0 && (
          <div>
            <span className="text-xs text-[var(--text-muted)]">Patterns</span>
            <p className="text-sm">{profile.patterns.join(", ")}</p>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {items.map((item) => (
          <div key={item.label} className="p-2 rounded-lg bg-[var(--bg-hover)]">
            <span className="text-xs text-[var(--text-muted)]">{item.label}</span>
            <p className={`font-mono text-sm ${item.warn ? "text-[var(--yellow)]" : ""}`}>
              {typeof item.value === "number" ? item.value.toFixed(2) : "N/A"}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

function SentimentBadge({ sentiment }: { sentiment: number }) {
  return (
    <div
      className="w-10 h-10 rounded-full flex items-center justify-center text-xs font-bold shrink-0"
      style={{
        backgroundColor: sentiment > 0.1 ? "rgba(34,197,94,0.2)" : sentiment < -0.1 ? "rgba(239,68,68,0.2)" : "rgba(136,136,160,0.2)",
        color: sentiment > 0.1 ? "var(--green)" : sentiment < -0.1 ? "var(--red)" : "var(--text-muted)",
      }}
    >
      {sentiment > 0 ? "+" : ""}
      {sentiment.toFixed(1)}
    </div>
  );
}

function NewsPanel({
  articles,
  sentimentScores,
}: {
  articles: PipelineResult["research"]["articles"];
  sentimentScores: PipelineResult["sentiment_profile"]["article_scores"];
}) {
  // Match scores to articles by url — titles aren't guaranteed unique
  // once real (non-mock) articles are in play.
  const scoreMap = new Map(sentimentScores.map((s) => [s.url, s]));

  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-5">
      <h3 className="text-sm font-medium text-[var(--text-muted)] mb-4">
        News Articles & Sentiment ({articles.length})
      </h3>
      <div className="space-y-3">
        {articles.map((a, i) => {
          const score = scoreMap.get(a.url);
          const sentiment = score?.sentiment ?? 0;

          return (
            <div key={i} className="p-3 rounded-lg bg-[var(--bg-hover)] flex gap-3">
              {/* Sentiment indicator */}
              <div className="flex flex-col items-center justify-center min-w-[48px]">
                <SentimentBadge sentiment={sentiment} />
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs px-1.5 py-0.5 rounded bg-[var(--bg-card)] text-[var(--text-muted)]">
                    {a.source}
                  </span>
                  <span className="text-xs px-1.5 py-0.5 rounded bg-[var(--bg-card)] text-[var(--purple)]">
                    {a.category}
                  </span>
                  {score && (
                    <span className="text-xs text-[var(--text-muted)]">
                      Impact: {score.expected_impact}
                    </span>
                  )}
                </div>
                {a.url ? (
                  <a
                    href={a.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm font-medium mb-1 block hover:text-[var(--blue)] hover:underline"
                  >
                    {a.title}
                  </a>
                ) : (
                  <p className="text-sm font-medium mb-1">{a.title}</p>
                )}
                <p className="text-xs text-[var(--text-muted)] line-clamp-2">{a.summary}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

interface NewsHistoryArticle {
  title: string;
  source: string;
  url: string;
  summary: string;
  category: string;
  published_at: string | null;
  relevance: number;
  sentiment: number | null;
  source_credibility: number | null;
  expected_impact: string | null;
  reasoning: string | null;
}

const NEWS_HISTORY_RANGES = [
  { label: "7D", days: 7 },
  { label: "30D", days: 30 },
  { label: "90D", days: 90 },
];

function NewsHistoryPanel({ ticker }: { ticker: string }) {
  const [days, setDays] = useState(30);
  const [articles, setArticles] = useState<NewsHistoryArticle[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setLoadError(null);
    fetch(`${API}/api/stocks/${ticker}/news?days=${days}`)
      .then((r) => r.json())
      .then((d) => setArticles(d.articles || []))
      .catch(() => setLoadError("Failed to load news history."))
      .finally(() => setLoading(false));
  }, [ticker, days]);

  return (
    <div className="mt-6">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <h2 className="text-lg font-semibold">News History</h2>
        <div className="flex gap-1">
          {NEWS_HISTORY_RANGES.map((r) => (
            <button
              key={r.label}
              onClick={() => setDays(r.days)}
              className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                days === r.days
                  ? "bg-[var(--bg-hover)] text-[var(--text)] border border-[var(--border)]"
                  : "text-[var(--text-muted)] hover:text-[var(--text)]"
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      {loading && <p className="text-sm text-[var(--text-muted)]">Loading…</p>}

      {!loading && loadError && <p className="text-sm text-[var(--red)]">{loadError}</p>}

      {!loading && !loadError && articles.length === 0 && (
        <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-6 text-center text-[var(--text-muted)]">
          <p className="text-sm">
            No persisted news yet for {ticker} in the last {days} days. Persisted history builds up
            each time the analysis pipeline runs with a Finnhub key configured — mock runs don&apos;t
            write here.
          </p>
        </div>
      )}

      {!loading && !loadError && articles.length > 0 && (
        <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-5">
          <p className="text-sm font-medium text-[var(--text-muted)] mb-4">
            {articles.length} article{articles.length === 1 ? "" : "s"} · last {days} days
          </p>
          <div className="space-y-3">
            {articles.map((a, i) => (
              <div key={i} className="p-3 rounded-lg bg-[var(--bg-hover)] flex gap-3">
                <div className="flex flex-col items-center justify-center min-w-[48px]">
                  <SentimentBadge sentiment={a.sentiment ?? 0} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <span className="text-xs px-1.5 py-0.5 rounded bg-[var(--bg-card)] text-[var(--text-muted)]">
                      {a.source}
                    </span>
                    <span className="text-xs px-1.5 py-0.5 rounded bg-[var(--bg-card)] text-[var(--purple)]">
                      {a.category}
                    </span>
                    {a.expected_impact && (
                      <span className="text-xs text-[var(--text-muted)]">Impact: {a.expected_impact}</span>
                    )}
                    {a.published_at && (
                      <span className="text-xs text-[var(--text-muted)] ml-auto">
                        {new Date(a.published_at).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                  {a.url ? (
                    <a
                      href={a.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm font-medium mb-1 block hover:text-[var(--blue)] hover:underline"
                    >
                      {a.title}
                    </a>
                  ) : (
                    <p className="text-sm font-medium mb-1">{a.title}</p>
                  )}
                  <p className="text-xs text-[var(--text-muted)] line-clamp-2">{a.summary}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ActionBadge({ action }: { action: string }) {
  const colors: Record<string, string> = {
    BUY: "bg-green-500/20 text-green-400 border-green-500/30",
    SELL: "bg-red-500/20 text-red-400 border-red-500/30",
    HOLD: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  };

  return (
    <span className={`text-xs font-bold px-2.5 py-1 rounded border ${colors[action] || ""}`}>
      {action}
    </span>
  );
}
