import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Dashboard from "./page";

// lightweight-charts renders onto a real <canvas> 2D context, which jsdom
// doesn't implement. We're testing that our own code drives the library's
// API correctly, not the library's rendering internals, so stub it out.
jest.mock("lightweight-charts", () => {
  const fakeSeries = () => ({
    setData: jest.fn(),
    applyOptions: jest.fn(),
    priceScale: jest.fn(() => ({ applyOptions: jest.fn() })),
  });
  const fakeChart = {
    addCandlestickSeries: jest.fn(fakeSeries),
    addAreaSeries: jest.fn(fakeSeries),
    addHistogramSeries: jest.fn(fakeSeries),
    removeSeries: jest.fn(),
    subscribeCrosshairMove: jest.fn(),
    applyOptions: jest.fn(),
    timeScale: jest.fn(() => ({ fitContent: jest.fn() })),
    remove: jest.fn(),
  };
  return {
    createChart: jest.fn(() => fakeChart),
    CrosshairMode: { Normal: 0 },
  };
});

function makeBar(close: number, daysAgo: number) {
  const ts = new Date(Date.now() - daysAgo * 86400000).toISOString();
  return { timestamp: ts, open: close, high: close * 1.01, low: close * 0.99, close, volume: 1000 };
}

const OHLCV_BODY = {
  ticker: "SPY",
  count: 2,
  data: [makeBar(340, 1), makeBar(345.11, 0)],
};

const PIPELINE_RESULT = {
  ticker: "SPY",
  elapsed_seconds: 0.9,
  signal: {
    action: "BUY",
    confidence: 62,
    reasoning: "Strong uptrend with confirming volume.",
    entry_low: 343.0,
    entry_high: 346.0,
    target: 355.0,
    stop_loss: 335.0,
    time_horizon: "2-4 weeks",
    risk_level: "moderate",
    composite_score: 0.4,
    factors: {},
    disclaimer: "For informational purposes only. Not financial advice.",
  },
  technical_profile: {
    trend_score: 0.5,
    momentum_score: 0.2,
    volatility_state: "contracting",
    volume_anomaly: false,
    patterns: [],
    indicators: {},
  },
  sentiment_profile: { composite_sentiment: 0.3, sentiment_trend: "improving", article_scores: [] },
  research: { articles: [] },
};

const PIPELINE_RESULT_WITH_NEWS = {
  ...PIPELINE_RESULT,
  sentiment_profile: {
    composite_sentiment: 0.3,
    sentiment_trend: "improving",
    article_scores: [
      {
        // Deliberately different title text from the article below (real
        // Finnhub titles can drift slightly from what was originally
        // fetched) — the join must go through `url`, not `title`.
        title: "Fed Signals Rate Cut (updated headline)",
        source: "Reuters",
        url: "https://reuters.com/mock/fed-rate-cut",
        sentiment: 0.7,
        expected_impact: "high",
      },
    ],
  },
  research: {
    articles: [
      {
        title: "Fed Signals Rate Cut",
        source: "Reuters",
        url: "https://reuters.com/mock/fed-rate-cut",
        summary: "The Fed hinted at a rate cut.",
        category: "macro",
        relevance: 0.9,
      },
    ],
  },
};

function mockFetchSequence(
  handlers: Record<string, (url: string, options?: RequestInit) => Promise<Response>>
) {
  global.fetch = jest.fn((url: string, options?: RequestInit) => {
    const match = Object.keys(handlers).find((key) => url.includes(key));
    if (!match) {
      return Promise.reject(new Error(`Unmocked fetch: ${url}`));
    }
    return handlers[match](url, options);
  }) as jest.Mock;
}

function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response);
}

describe("Dashboard", () => {
  afterEach(() => {
    // clearAllMocks (not resetAllMocks) — resetting would also wipe the
    // lightweight-charts mock's implementation set up in jest.mock() above.
    jest.clearAllMocks();
    // The Dashboard persists the selected ticker to localStorage, which
    // jsdom does not reset between tests on its own — without this, a
    // ticker switch in one test leaks into the next test's initial render.
    window.localStorage.clear();
  });

  it("loads OHLCV data and renders the latest price", async () => {
    mockFetchSequence({
      "/ohlcv": () => jsonResponse(OHLCV_BODY),
      "/api/signals/SPY": () => jsonResponse({ signals: [] }),
      "/api/stocks/SPY/news": () => jsonResponse({ ticker: "SPY", count: 0, articles: [] }),
      "/api/watchlist/summary": () =>
        jsonResponse({ tickers: [{ ticker: "SPY", price: null, signal: null }] }),
    });

    render(<Dashboard />);

    expect(await screen.findAllByText("$345.11")).not.toHaveLength(0);
    expect(screen.getByText(/No signals yet/i)).toBeInTheDocument();
  });

  it("switches ticker and refetches its data when a watchlist row is clicked", async () => {
    const AAPL_OHLCV = { ticker: "AAPL", count: 2, data: [makeBar(180, 1), makeBar(185, 0)] };
    mockFetchSequence({
      "/api/watchlist/summary": () =>
        jsonResponse({
          tickers: [
            { ticker: "SPY", price: null, signal: null },
            { ticker: "AAPL", price: null, signal: null },
          ],
        }),
      "/api/stocks/AAPL/ohlcv": () => jsonResponse(AAPL_OHLCV),
      "/api/signals/AAPL": () => jsonResponse({ signals: [] }),
      "/api/stocks/AAPL/news": () => jsonResponse({ ticker: "AAPL", count: 0, articles: [] }),
      "/ohlcv": () => jsonResponse(OHLCV_BODY),
      "/api/signals/SPY": () => jsonResponse({ signals: [] }),
      "/api/stocks/SPY/news": () => jsonResponse({ ticker: "SPY", count: 0, articles: [] }),
    });

    const user = userEvent.setup();
    render(<Dashboard />);

    expect(await screen.findAllByText("$345.11")).not.toHaveLength(0);
    await user.click(await screen.findByRole("button", { name: "AAPL" }));

    expect(await screen.findAllByText("$185.00")).not.toHaveLength(0);
  });

  it("clears a stale sync confirmation message when switching tickers", async () => {
    const AAPL_OHLCV = { ticker: "AAPL", count: 2, data: [makeBar(180, 1), makeBar(185, 0)] };
    mockFetchSequence({
      "/api/watchlist/summary": () =>
        jsonResponse({
          tickers: [
            { ticker: "SPY", price: null, signal: null },
            { ticker: "AAPL", price: null, signal: null },
          ],
        }),
      "/api/stocks/SPY/sync": () =>
        jsonResponse({ ticker: "SPY", interval: "daily", mode: "incremental", synced_bars: 251, latest_close: 769.06 }),
      "/api/stocks/AAPL/ohlcv": () => jsonResponse(AAPL_OHLCV),
      "/api/signals/AAPL": () => jsonResponse({ signals: [] }),
      "/api/stocks/AAPL/news": () => jsonResponse({ ticker: "AAPL", count: 0, articles: [] }),
      "/ohlcv": () => jsonResponse(OHLCV_BODY),
      "/api/signals/SPY": () => jsonResponse({ signals: [] }),
      "/api/stocks/SPY/news": () => jsonResponse({ ticker: "SPY", count: 0, articles: [] }),
    });

    const user = userEvent.setup();
    render(<Dashboard />);

    await screen.findAllByText("$345.11");
    await user.click(screen.getByRole("button", { name: "Sync Live Data" }));

    expect(await screen.findByText(/Synced 251 bars/i)).toBeInTheDocument();

    await user.click(await screen.findByRole("button", { name: "AAPL" }));
    await screen.findAllByText("$185.00");

    expect(screen.queryByText(/Synced 251 bars/i)).not.toBeInTheDocument();
  });

  it("restores the last-selected ticker across a remount (simulating a page reload)", async () => {
    const AAPL_OHLCV = { ticker: "AAPL", count: 2, data: [makeBar(180, 1), makeBar(185, 0)] };
    const handlers = {
      "/api/watchlist/summary": () =>
        jsonResponse({
          tickers: [
            { ticker: "SPY", price: null, signal: null },
            { ticker: "AAPL", price: null, signal: null },
          ],
        }),
      "/api/stocks/AAPL/ohlcv": () => jsonResponse(AAPL_OHLCV),
      "/api/signals/AAPL": () => jsonResponse({ signals: [] }),
      "/api/stocks/AAPL/news": () => jsonResponse({ ticker: "AAPL", count: 0, articles: [] }),
      "/ohlcv": () => jsonResponse(OHLCV_BODY),
      "/api/signals/SPY": () => jsonResponse({ signals: [] }),
      "/api/stocks/SPY/news": () => jsonResponse({ ticker: "SPY", count: 0, articles: [] }),
    };
    mockFetchSequence(handlers);

    const user = userEvent.setup();
    const { unmount } = render(<Dashboard />);
    await screen.findAllByText("$345.11");
    await user.click(await screen.findByRole("button", { name: "AAPL" }));
    await screen.findAllByText("$185.00");
    unmount();

    // A fresh mount reads the persisted ticker from localStorage — this is
    // the client-only effect run, so the component briefly renders the
    // SPY default before switching, same as it would after a real reload.
    mockFetchSequence(handlers);
    render(<Dashboard />);

    expect(await screen.findAllByText("$185.00")).not.toHaveLength(0);
  });

  it("adds a new ticker via the search box, auto-syncs real data for it, and switches to it", async () => {
    const syncCalls: string[] = [];
    mockFetchSequence({
      "/api/watchlist/summary": () =>
        jsonResponse({ tickers: [{ ticker: "SPY", price: null, signal: null }] }),
      "/api/watchlist/": (url, options) => {
        if (options?.method === "POST") {
          return jsonResponse({ status: "added", ticker: "TSLA" });
        }
        return Promise.reject(new Error(`Unmocked fetch: ${url}`));
      },
      "/api/stocks/TSLA/sync": (url, options) => {
        syncCalls.push(`${options?.method} ${url}`);
        return jsonResponse({ ticker: "TSLA", interval: "daily", mode: "full", synced_bars: 250, latest_close: 250.5 });
      },
      "/api/stocks/TSLA/ohlcv": () => jsonResponse({ ticker: "TSLA", count: 0, data: [] }),
      "/api/signals/TSLA": () => jsonResponse({ signals: [] }),
      "/api/stocks/TSLA/news": () => jsonResponse({ ticker: "TSLA", count: 0, articles: [] }),
      "/ohlcv": () => jsonResponse(OHLCV_BODY),
      "/api/signals/SPY": () => jsonResponse({ signals: [] }),
      "/api/stocks/SPY/news": () => jsonResponse({ ticker: "SPY", count: 0, articles: [] }),
    });

    const user = userEvent.setup();
    render(<Dashboard />);

    await screen.findAllByText("$345.11");
    await user.type(screen.getByPlaceholderText("Search or add ticker"), "tsla");
    await user.click(await screen.findByRole("button", { name: /add "tsla" to watchlist/i }));

    expect(await screen.findByRole("button", { name: "TSLA" })).toBeInTheDocument();
    expect(syncCalls).toEqual(["POST http://localhost:8000/api/stocks/TSLA/sync?interval=daily"]);
  });

  it("still adds and switches to the ticker even if the auto-sync request fails", async () => {
    mockFetchSequence({
      "/api/watchlist/summary": () =>
        jsonResponse({ tickers: [{ ticker: "SPY", price: null, signal: null }] }),
      "/api/watchlist/": (url, options) => {
        if (options?.method === "POST") {
          return jsonResponse({ status: "added", ticker: "TSLA" });
        }
        return Promise.reject(new Error(`Unmocked fetch: ${url}`));
      },
      "/api/stocks/TSLA/sync": () => Promise.reject(new Error("upstream unavailable")),
      "/api/stocks/TSLA/ohlcv": () => jsonResponse({ ticker: "TSLA", count: 0, data: [] }),
      "/api/signals/TSLA": () => jsonResponse({ signals: [] }),
      "/api/stocks/TSLA/news": () => jsonResponse({ ticker: "TSLA", count: 0, articles: [] }),
      "/ohlcv": () => jsonResponse(OHLCV_BODY),
      "/api/signals/SPY": () => jsonResponse({ signals: [] }),
      "/api/stocks/SPY/news": () => jsonResponse({ ticker: "SPY", count: 0, articles: [] }),
    });

    const user = userEvent.setup();
    render(<Dashboard />);

    await screen.findAllByText("$345.11");
    await user.type(screen.getByPlaceholderText("Search or add ticker"), "tsla");
    await user.click(await screen.findByRole("button", { name: /add "tsla" to watchlist/i }));

    expect(await screen.findByRole("button", { name: "TSLA" })).toBeInTheDocument();
    expect(await screen.findByText(/No daily data yet for TSLA/i)).toBeInTheDocument();
  });

  it("does not allow removing the only ticker in the watchlist", async () => {
    mockFetchSequence({
      "/api/watchlist/summary": () =>
        jsonResponse({ tickers: [{ ticker: "SPY", price: null, signal: null }] }),
      "/ohlcv": () => jsonResponse(OHLCV_BODY),
      "/api/signals/SPY": () => jsonResponse({ signals: [] }),
      "/api/stocks/SPY/news": () => jsonResponse({ ticker: "SPY", count: 0, articles: [] }),
    });

    render(<Dashboard />);

    await screen.findAllByText("$345.11");
    expect(screen.queryByRole("button", { name: /remove spy/i })).not.toBeInTheDocument();
  });

  it("filters the watchlist overview by signal action", async () => {
    mockFetchSequence({
      "/api/watchlist/summary": () =>
        jsonResponse({
          tickers: [
            { ticker: "SPY", price: 500, signal: { action: "BUY", confidence: 60, timestamp: "2026-08-30T00:00:00" } },
            { ticker: "AAPL", price: 200, signal: { action: "SELL", confidence: 40, timestamp: "2026-08-30T00:00:00" } },
            { ticker: "MSFT", price: 300, signal: null },
          ],
        }),
      "/ohlcv": () => jsonResponse(OHLCV_BODY),
      "/api/signals/SPY": () => jsonResponse({ signals: [] }),
      "/api/stocks/SPY/news": () => jsonResponse({ ticker: "SPY", count: 0, articles: [] }),
    });

    const user = userEvent.setup();
    render(<Dashboard />);

    await screen.findAllByText("$345.11");
    expect(await screen.findByRole("button", { name: "AAPL" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "MSFT" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "BUY (1)" }));

    expect(screen.getByRole("button", { name: "SPY" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "AAPL" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "MSFT" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "No Signal (1)" }));
    expect(screen.getByRole("button", { name: "MSFT" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "SPY" })).not.toBeInTheDocument();
  });

  it("sorts the watchlist overview by ticker, price, and confidence", async () => {
    mockFetchSequence({
      "/api/watchlist/summary": () =>
        jsonResponse({
          tickers: [
            { ticker: "ZETA", price: 50, signal: { action: "HOLD", confidence: 10, timestamp: "2026-08-30T00:00:00" } },
            { ticker: "ALPHA", price: 200, signal: { action: "HOLD", confidence: 90, timestamp: "2026-08-30T00:00:00" } },
            { ticker: "MID", price: 100, signal: { action: "HOLD", confidence: 50, timestamp: "2026-08-30T00:00:00" } },
          ],
        }),
      "/ohlcv": () => jsonResponse(OHLCV_BODY),
      "/api/signals/SPY": () => jsonResponse({ signals: [] }),
      "/api/stocks/SPY/news": () => jsonResponse({ ticker: "SPY", count: 0, articles: [] }),
    });

    const user = userEvent.setup();
    render(<Dashboard />);
    await screen.findAllByText("$345.11");
    await screen.findByRole("button", { name: "ALPHA" });

    const tickerOrder = () =>
      screen
        .getAllByRole("button")
        .map((b) => b.textContent)
        .filter((t): t is string => t === "ZETA" || t === "ALPHA" || t === "MID");

    // Default sort: ticker ascending
    expect(tickerOrder()).toEqual(["ALPHA", "MID", "ZETA"]);

    await user.click(screen.getByRole("button", { name: "Sort by Price" }));
    expect(tickerOrder()).toEqual(["ZETA", "MID", "ALPHA"]); // ascending price: 50, 100, 200

    await user.click(screen.getByRole("button", { name: "Sort by Price" }));
    expect(tickerOrder()).toEqual(["ALPHA", "MID", "ZETA"]); // descending price: 200, 100, 50

    await user.click(screen.getByRole("button", { name: "Sort by Confidence" }));
    expect(tickerOrder()).toEqual(["ZETA", "MID", "ALPHA"]); // ascending confidence: 10, 50, 90
    expect(screen.queryByRole("button", { name: "SPY" })).not.toBeInTheDocument();
  });

  it("renders persisted news history independent of running the pipeline", async () => {
    mockFetchSequence({
      "/ohlcv": () => jsonResponse(OHLCV_BODY),
      "/api/signals/SPY": () => jsonResponse({ signals: [] }),
      "/api/stocks/SPY/news": () =>
        jsonResponse({
          ticker: "SPY",
          count: 1,
          articles: [
            {
              title: "Fed Signals Rate Cut",
              source: "Reuters",
              url: "https://reuters.com/mock/fed-rate-cut",
              summary: "The Fed hinted at a rate cut.",
              category: "macro",
              published_at: "2026-08-10T12:00:00",
              relevance: 0.9,
              sentiment: 0.7,
              source_credibility: 0.95,
              expected_impact: "high",
              reasoning: "Dovish signal",
            },
          ],
        }),
    });

    render(<Dashboard />);

    expect(await screen.findByText("News History")).toBeInTheDocument();
    const link = await screen.findByRole("link", { name: /fed signals rate cut/i });
    expect(link).toHaveAttribute("href", "https://reuters.com/mock/fed-rate-cut");
    expect(screen.getByText("+0.7")).toBeInTheDocument();
  });

  it("shows an empty state for news history when nothing is persisted yet", async () => {
    mockFetchSequence({
      "/ohlcv": () => jsonResponse(OHLCV_BODY),
      "/api/signals/SPY": () => jsonResponse({ signals: [] }),
      "/api/stocks/SPY/news": () => jsonResponse({ ticker: "SPY", count: 0, articles: [] }),
    });

    render(<Dashboard />);

    expect(await screen.findByText(/No persisted news yet/i)).toBeInTheDocument();
  });

  it("runs the pipeline and renders the resulting signal", async () => {
    mockFetchSequence({
      "/ohlcv": () => jsonResponse(OHLCV_BODY),
      "/api/signals/SPY": () => jsonResponse({ signals: [] }),
      "/api/pipeline/run/SPY": () => jsonResponse(PIPELINE_RESULT),
    });

    const user = userEvent.setup();
    render(<Dashboard />);

    await screen.findAllByText("$345.11");
    await user.click(screen.getByRole("button", { name: /run analysis pipeline/i }));

    expect(await screen.findByText("BUY")).toBeInTheDocument();
    expect(screen.getByText(/Strong uptrend with confirming volume/i)).toBeInTheDocument();
  });

  it("links each news article to its source URL", async () => {
    mockFetchSequence({
      "/ohlcv": () => jsonResponse(OHLCV_BODY),
      "/api/signals/SPY": () => jsonResponse({ signals: [] }),
      "/api/pipeline/run/SPY": () => jsonResponse(PIPELINE_RESULT_WITH_NEWS),
    });

    const user = userEvent.setup();
    render(<Dashboard />);

    await screen.findAllByText("$345.11");
    await user.click(screen.getByRole("button", { name: /run analysis pipeline/i }));
    await screen.findByText("BUY");

    await user.click(screen.getByRole("button", { name: /news & sentiment/i }));

    const link = await screen.findByRole("link", { name: /fed signals rate cut/i });
    expect(link).toHaveAttribute("href", "https://reuters.com/mock/fed-rate-cut");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", expect.stringContaining("noopener"));

    // The sentiment score's title deliberately differs from the article's
    // own title above — this only renders if the join matched on `url`.
    expect(screen.getByText("+0.7")).toBeInTheDocument();
  });

  it("shows a status-coded error when the pipeline responds with a server error", async () => {
    mockFetchSequence({
      "/ohlcv": () => jsonResponse(OHLCV_BODY),
      "/api/signals/SPY": () => jsonResponse({ signals: [] }),
      "/api/pipeline/run/SPY": () => jsonResponse({}, 500),
    });

    const user = userEvent.setup();
    render(<Dashboard />);

    await screen.findAllByText("$345.11");
    await user.click(screen.getByRole("button", { name: /run analysis pipeline/i }));

    expect(await screen.findByText(/Pipeline failed \(500\)/i)).toBeInTheDocument();
  });

  it("shows a connectivity error when the backend is unreachable", async () => {
    global.fetch = jest.fn((url: string) => {
      if (url.includes("/api/pipeline/run/SPY")) {
        return Promise.reject(new Error("network down"));
      }
      if (url.includes("/ohlcv")) return jsonResponse(OHLCV_BODY);
      if (url.includes("/api/signals/SPY")) return jsonResponse({ signals: [] });
      return Promise.reject(new Error(`Unmocked fetch: ${url}`));
    }) as jest.Mock;

    const user = userEvent.setup();
    render(<Dashboard />);

    await screen.findAllByText("$345.11");
    await user.click(screen.getByRole("button", { name: /run analysis pipeline/i }));

    expect(await screen.findByText(/Is the backend running/i)).toBeInTheDocument();
  });
});
