import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Dashboard from "./page";

// lightweight-charts renders onto a real <canvas> 2D context, which jsdom
// doesn't implement. We're testing that our own code drives the library's
// API correctly, not the library's rendering internals, so stub it out.
jest.mock("lightweight-charts", () => {
  const fakeSeries = () => ({
    setData: jest.fn(),
    priceScale: jest.fn(() => ({ applyOptions: jest.fn() })),
  });
  const fakeChart = {
    addCandlestickSeries: jest.fn(fakeSeries),
    addHistogramSeries: jest.fn(fakeSeries),
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

function mockFetchSequence(handlers: Record<string, () => Promise<Response>>) {
  global.fetch = jest.fn((url: string) => {
    const match = Object.keys(handlers).find((key) => url.includes(key));
    if (!match) {
      return Promise.reject(new Error(`Unmocked fetch: ${url}`));
    }
    return handlers[match]();
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
  });

  it("loads OHLCV data and renders the latest price", async () => {
    mockFetchSequence({
      "/ohlcv": () => jsonResponse(OHLCV_BODY),
      "/api/signals/SPY": () => jsonResponse({ signals: [] }),
      "/api/stocks/SPY/news": () => jsonResponse({ ticker: "SPY", count: 0, articles: [] }),
    });

    render(<Dashboard />);

    expect(await screen.findByText("$345.11")).toBeInTheDocument();
    expect(screen.getByText(/No signals yet/i)).toBeInTheDocument();
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

    await screen.findByText("$345.11");
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

    await screen.findByText("$345.11");
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

    await screen.findByText("$345.11");
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

    await screen.findByText("$345.11");
    await user.click(screen.getByRole("button", { name: /run analysis pipeline/i }));

    expect(await screen.findByText(/Is the backend running/i)).toBeInTheDocument();
  });
});
