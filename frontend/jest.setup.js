import "@testing-library/jest-dom";

// jsdom doesn't implement canvas; components that check for a null context
// and no-op are fine, so stub it out to avoid noisy "not implemented" errors.
HTMLCanvasElement.prototype.getContext = jest.fn(() => null);

// lightweight-charts needs both of these to construct a chart; jsdom has
// neither.
window.matchMedia =
  window.matchMedia ||
  function matchMedia(query) {
    return {
      matches: false,
      media: query,
      onchange: null,
      addListener: jest.fn(),
      removeListener: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    };
  };

global.ResizeObserver =
  global.ResizeObserver ||
  class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
