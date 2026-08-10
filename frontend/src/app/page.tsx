"use client";

import { useState, useEffect, useRef, useCallback } from "react";

const API = "http://localhost:8000";

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
      sentiment: number;
      expected_impact: string;
    }>;
  };
  research: {
    articles: Array<{
      title: string;
      source: string;
      summary: string;
      category: string;
      relevance: number;
    }>;
  };
}

// ── Main Dashboard ──

export default function Dashboard() {
  const [ticker] = useState("SPY");
  const [ohlcv, setOhlcv] = useState<OHLCVBar[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [pipelineResult, setPipelineResult] = useState<PipelineResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"signal" | "technical" | "news">("signal");

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
          <PriceChart data={ohlcv} />
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
    </div>
  );
}

// ── Components ──

function PriceChart({ data }: { data: OHLCVBar[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || data.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const padding = { top: 20, right: 60, bottom: 30, left: 10 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    // Use last 120 bars for visibility
    const visible = data.slice(-120);
    const closes = visible.map((d) => d.close);
    const minP = Math.min(...visible.map((d) => d.low)) * 0.998;
    const maxP = Math.max(...visible.map((d) => d.high)) * 1.002;

    const xScale = (i: number) => padding.left + (i / (visible.length - 1)) * chartW;
    const yScale = (p: number) => padding.top + (1 - (p - minP) / (maxP - minP)) * chartH;

    // Clear
    ctx.fillStyle = "#12121a";
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = "#1a1a2e";
    ctx.lineWidth = 0.5;
    const gridSteps = 5;
    for (let i = 0; i <= gridSteps; i++) {
      const y = padding.top + (i / gridSteps) * chartH;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();

      // Price labels
      const price = maxP - (i / gridSteps) * (maxP - minP);
      ctx.fillStyle = "#8888a0";
      ctx.font = "11px system-ui";
      ctx.textAlign = "left";
      ctx.fillText(`$${price.toFixed(0)}`, w - padding.right + 5, y + 4);
    }

    // Candlesticks
    const barWidth = Math.max(1, chartW / visible.length * 0.6);
    visible.forEach((bar, i) => {
      const x = xScale(i);
      const isGreen = bar.close >= bar.open;
      ctx.fillStyle = isGreen ? "#22c55e" : "#ef4444";
      ctx.strokeStyle = isGreen ? "#22c55e" : "#ef4444";

      // Wick
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, yScale(bar.high));
      ctx.lineTo(x, yScale(bar.low));
      ctx.stroke();

      // Body
      const bodyTop = yScale(Math.max(bar.open, bar.close));
      const bodyBot = yScale(Math.min(bar.open, bar.close));
      const bodyH = Math.max(1, bodyBot - bodyTop);
      ctx.fillRect(x - barWidth / 2, bodyTop, barWidth, bodyH);
    });

    // Volume bars at bottom
    const maxVol = Math.max(...visible.map((d) => d.volume));
    const volH = chartH * 0.15;
    visible.forEach((bar, i) => {
      const x = xScale(i);
      const isGreen = bar.close >= bar.open;
      ctx.fillStyle = isGreen ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)";
      const bh = (bar.volume / maxVol) * volH;
      ctx.fillRect(x - barWidth / 2, padding.top + chartH - bh, barWidth, bh);
    });
  }, [data]);

  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-4">
      <h2 className="text-sm font-medium text-[var(--text-muted)] mb-2">Price Chart (Last 120 days)</h2>
      <canvas ref={canvasRef} className="w-full" style={{ height: 350 }} />
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

function NewsPanel({
  articles,
  sentimentScores,
}: {
  articles: PipelineResult["research"]["articles"];
  sentimentScores: PipelineResult["sentiment_profile"]["article_scores"];
}) {
  // Match scores to articles by title
  const scoreMap = new Map(sentimentScores.map((s) => [s.title, s]));

  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-card)] p-5">
      <h3 className="text-sm font-medium text-[var(--text-muted)] mb-4">
        News Articles & Sentiment ({articles.length})
      </h3>
      <div className="space-y-3">
        {articles.map((a, i) => {
          const score = scoreMap.get(a.title);
          const sentiment = score?.sentiment ?? 0;

          return (
            <div key={i} className="p-3 rounded-lg bg-[var(--bg-hover)] flex gap-3">
              {/* Sentiment indicator */}
              <div className="flex flex-col items-center justify-center min-w-[48px]">
                <div
                  className="w-10 h-10 rounded-full flex items-center justify-center text-xs font-bold"
                  style={{
                    backgroundColor: sentiment > 0.1 ? "rgba(34,197,94,0.2)" : sentiment < -0.1 ? "rgba(239,68,68,0.2)" : "rgba(136,136,160,0.2)",
                    color: sentiment > 0.1 ? "var(--green)" : sentiment < -0.1 ? "var(--red)" : "var(--text-muted)",
                  }}
                >
                  {sentiment > 0 ? "+" : ""}
                  {sentiment.toFixed(1)}
                </div>
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
                <p className="text-sm font-medium mb-1">{a.title}</p>
                <p className="text-xs text-[var(--text-muted)] line-clamp-2">{a.summary}</p>
              </div>
            </div>
          );
        })}
      </div>
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
