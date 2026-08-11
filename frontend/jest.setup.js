import "@testing-library/jest-dom";

// jsdom doesn't implement canvas; the price chart checks for a null context
// and no-ops, so stub it out to avoid noisy "not implemented" console errors.
HTMLCanvasElement.prototype.getContext = jest.fn(() => null);
