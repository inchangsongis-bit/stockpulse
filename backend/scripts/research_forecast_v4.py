"""
Final research pass — the decisive question.

v3 produced three results that together point one direction:
  * The signal exists ONLY at a 5-minute horizon (AUC 0.524) and is gone
    by 15 minutes (0.5015), 30 (0.4999), 60 (0.5011), 120 (0.4961).
  * Filtering OUT thin extended-hours bars made the model WORSE, not
    better — the opposite of what removing noise should do.
  * Extended-hours 1-min returns have ~9x stronger negative
    autocorrelation than regular hours (-0.2075 vs -0.0229).

Signal that lives only at the shortest horizon, dies immediately beyond
it, and is strongest where the bid-ask bounce is strongest, is the
textbook signature of MICROSTRUCTURE NOISE rather than real predictive
alpha. A model can learn bid-ask bounce — price alternating between bid
and ask produces mechanical, learnable mean reversion — but it cannot be
traded, because capturing it means crossing the spread every round trip.

This script is the test that settles it: require the 5-minute move to
exceed a realistic round-trip transaction cost before it counts as
up/down, and see whether any predictive edge survives. Regular trading
hours only, because that's when you could actually execute.

If accuracy collapses to ~50% once costs are imposed, the honest
conclusion is that the headline 51.9% is an artifact and there is no
tradeable 5-minute edge in OHLCV data alone.

Run: cd backend && source venv/bin/activate && python scripts/research_forecast_v4.py
"""

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.forecast_features import FEATURE_COLUMNS as BASE_FEATURES, build_features  # noqa: E402
from database import async_session  # noqa: E402
from models import OHLCV  # noqa: E402

RTH_START_MIN, RTH_END_MIN = 390, 780
HORIZON = 5

# Round-trip cost thresholds in basis points. A large-cap US equity spread
# is typically ~1-3bps; 5bps is a realistic all-in round trip for a retail
# marketable order, 10bps a conservative one.
COST_THRESHOLDS_BPS = [0, 1, 2, 5, 10]


async def load_minute_bars_by_ticker() -> dict:
    async with async_session() as db:
        result = await db.execute(
            select(OHLCV).where(OHLCV.interval == "minute").order_by(OHLCV.ticker, OHLCV.timestamp.asc())
        )
        rows = result.scalars().all()
    bars = defaultdict(list)
    for r in rows:
        bars[r.ticker].append({
            "timestamp": r.timestamp, "open": r.open, "high": r.high,
            "low": r.low, "close": r.close, "volume": r.volume,
        })
    return {t: pd.DataFrame(b) for t, b in bars.items()}


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    mod = ts.dt.hour * 60 + ts.dt.minute
    return df[(mod >= RTH_START_MIN) & (mod <= RTH_END_MIN)].reset_index(drop=True)


def cost_aware_labels(df: pd.DataFrame, cost_bps: float) -> pd.Series:
    """1 if the 5-min forward return clears +cost, 0 if it clears -cost,
    NaN in between (a move too small to trade profitably) or across a
    session gap."""
    close = df["close"]
    ts = pd.to_datetime(df["timestamp"])
    fwd_ret = (close.shift(-HORIZON) - close) / close
    gap = (ts.shift(-HORIZON) - ts).dt.total_seconds() / 60
    thresh = cost_bps / 10000.0

    label = pd.Series(np.nan, index=df.index)
    label[fwd_ret > thresh] = 1.0
    label[fwd_ret < -thresh] = 0.0
    label[gap != HORIZON] = np.nan
    return label


def main():
    print("Loading minute bars...")
    bars_by_ticker = asyncio.run(load_minute_bars_by_ticker())
    bars_by_ticker = {t: filter_rth(b) for t, b in bars_by_ticker.items()}
    bars_by_ticker = {t: b for t, b in bars_by_ticker.items() if len(b) > 200}
    print(f"  {len(bars_by_ticker)} tickers, {sum(len(b) for b in bars_by_ticker.values())} "
          f"regular-hours bars\n")

    print("=" * 98)
    print("K — Does any edge survive realistic transaction costs? (regular hours, 5-min horizon)")
    print("=" * 98)
    print(f"  {'cost threshold':<20s}{'tradeable rows':>16s}{'coverage':>11s}{'accuracy':>11s}{'AUC':>9s}")

    # Total labelable rows at 0bps, as the coverage denominator.
    baseline_n = None

    for cost_bps in COST_THRESHOLDS_BPS:
        tr_X, tr_y, te_X, te_y = [], [], [], []
        for ticker, bars in bars_by_ticker.items():
            featured = build_features(bars)
            labels = cost_aware_labels(bars, cost_bps)
            valid = featured[BASE_FEATURES].notna().all(axis=1) & labels.notna()
            X = featured.loc[valid, BASE_FEATURES].reset_index(drop=True)
            y = labels.loc[valid].reset_index(drop=True)
            if len(X) < 30 or y.nunique() < 2:
                continue
            cut = int(len(X) * 0.8)
            tr_X.append(X.iloc[:cut]); tr_y.append(y.iloc[:cut])
            te_X.append(X.iloc[cut:]); te_y.append(y.iloc[cut:])

        X_train = pd.concat(tr_X, ignore_index=True); y_train = pd.concat(tr_y, ignore_index=True)
        X_test = pd.concat(te_X, ignore_index=True); y_test = pd.concat(te_y, ignore_index=True)
        n_total = len(X_train) + len(X_test)
        if baseline_n is None:
            baseline_n = n_total

        model = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = float("nan")

        label = f"{cost_bps}bps round trip" if cost_bps else "0bps (no cost)"
        print(f"  {label:<20s}{n_total:>16d}{n_total / baseline_n:>10.1%}{acc:>11.4f}{auc:>9.4f}")

    print("\n" + "=" * 98)
    print("L — For reference: how big IS a typical 5-minute move, vs. the cost of trading it?")
    print("=" * 98)
    moves = []
    for ticker, bars in bars_by_ticker.items():
        close = bars["close"]
        ts = pd.to_datetime(bars["timestamp"])
        fwd = (close.shift(-HORIZON) - close) / close
        gap = (ts.shift(-HORIZON) - ts).dt.total_seconds() / 60
        moves.append(fwd[gap == HORIZON].abs().dropna())
    all_moves = pd.concat(moves) * 10000  # to basis points
    print(f"  Median |5-min move| : {all_moves.median():>7.2f} bps")
    print(f"  Mean   |5-min move| : {all_moves.mean():>7.2f} bps")
    print(f"  25th percentile     : {all_moves.quantile(0.25):>7.2f} bps")
    print(f"  75th percentile     : {all_moves.quantile(0.75):>7.2f} bps")
    print(f"\n  Share of 5-min moves smaller than a 5bps round trip: "
          f"{(all_moves < 5).mean():.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
