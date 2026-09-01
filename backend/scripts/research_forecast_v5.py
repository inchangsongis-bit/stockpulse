"""
The number that actually settles it: simulated net P&L.

v4 showed the ~51.5% edge SURVIVES cost-aware relabeling at every
threshold (AUC ~0.52 even when only moves >10bps count), so it isn't
purely bid-ask bounce — there is a small genuine short-horizon effect.
But "can predict direction of moves bigger than X" is not the same
question as "makes money." This script answers the second one.

For each configuration it simulates actually taking the model's trade:
enter at the current close, exit 5 minutes later, pay a round-trip cost,
and report the net basis points per trade. It also computes the
BREAK-EVEN accuracy — the win rate you'd need, given the observed average
move size, for the strategy to clear its own costs. Comparing that to the
accuracy we actually have is the whole answer.

Also evaluated: trading only the model's highest-confidence predictions,
since v2 found accuracy rises monotonically with confidence (49.6% in the
least-confident decile to 52.8% in the most). If any configuration is
profitable, it would be that one.

Run: cd backend && source venv/bin/activate && python scripts/research_forecast_v5.py
"""

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.forecast_features import FEATURE_COLUMNS as BASE_FEATURES, build_features  # noqa: E402
from database import async_session  # noqa: E402
from models import OHLCV  # noqa: E402

RTH_START_MIN, RTH_END_MIN = 390, 780
HORIZON = 5
COST_SCENARIOS_BPS = [1.0, 2.0, 5.0]


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


def main():
    print("Loading regular-hours minute bars...")
    bars_by_ticker = asyncio.run(load_minute_bars_by_ticker())
    bars_by_ticker = {t: filter_rth(b) for t, b in bars_by_ticker.items()}
    bars_by_ticker = {t: b for t, b in bars_by_ticker.items() if len(b) > 200}
    print(f"  {len(bars_by_ticker)} tickers\n")

    # Build train/test with the FORWARD RETURN retained alongside the label,
    # so the test set can be turned into a P&L simulation rather than just
    # an accuracy score.
    tr_X, tr_y, te_X, te_y, te_ret = [], [], [], [], []
    for ticker, bars in bars_by_ticker.items():
        featured = build_features(bars)
        close = bars["close"]
        ts = pd.to_datetime(bars["timestamp"])
        fwd_ret = (close.shift(-HORIZON) - close) / close
        gap = (ts.shift(-HORIZON) - ts).dt.total_seconds() / 60
        fwd_ret[gap != HORIZON] = np.nan
        labels = (fwd_ret > 0).astype(float)
        labels[fwd_ret.isna()] = np.nan

        valid = featured[BASE_FEATURES].notna().all(axis=1) & labels.notna()
        X = featured.loc[valid, BASE_FEATURES].reset_index(drop=True)
        y = labels.loc[valid].reset_index(drop=True)
        r = fwd_ret.loc[valid].reset_index(drop=True)
        if len(X) < 30:
            continue
        cut = int(len(X) * 0.8)
        tr_X.append(X.iloc[:cut]); tr_y.append(y.iloc[:cut])
        te_X.append(X.iloc[cut:]); te_y.append(y.iloc[cut:]); te_ret.append(r.iloc[cut:])

    X_train = pd.concat(tr_X, ignore_index=True); y_train = pd.concat(tr_y, ignore_index=True)
    X_test = pd.concat(te_X, ignore_index=True); y_test = pd.concat(te_y, ignore_index=True)
    ret_test = pd.concat(te_ret, ignore_index=True).to_numpy() * 10000  # basis points

    print(f"Training on {len(X_train)} rows, simulating on {len(X_test)} held-out rows...\n")
    model = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    def simulate(mask, name):
        p = proba[mask]
        r = ret_test[mask]
        # Long when P(up) >= 0.5, short otherwise. Gross P&L per trade is
        # the forward move signed by the direction we took.
        direction = np.where(p >= 0.5, 1.0, -1.0)
        gross_bps = (direction * r).mean()
        win_rate = (np.sign(direction) == np.sign(r)).mean()
        avg_abs_move = np.abs(r).mean()
        # Break-even win rate: gross expectation with win rate w on a move
        # of size m is m*(2w-1); set equal to cost and solve for w.
        print(f"  {name}")
        print(f"    trades              : {mask.sum():>10,d}")
        print(f"    win rate            : {win_rate:>10.2%}")
        print(f"    avg |5-min move|    : {avg_abs_move:>10.2f} bps")
        print(f"    gross P&L per trade : {gross_bps:>+10.3f} bps")
        for cost in COST_SCENARIOS_BPS:
            net = gross_bps - cost
            verdict = "PROFITABLE" if net > 0 else "loses money"
            print(f"      net @ {cost:>4.1f}bps cost : {net:>+8.3f} bps   -> {verdict}")
        for cost in COST_SCENARIOS_BPS:
            be = (cost / avg_abs_move + 1) / 2
            print(f"      break-even win rate @ {cost:>4.1f}bps cost : {be:>6.2%}")
        print()

    print("=" * 92)
    print("M — Net P&L simulation on the held-out set")
    print("=" * 92)

    simulate(np.ones(len(proba), dtype=bool), "ALL predictions")

    conf = np.abs(proba - 0.5)
    for pct, name in ((90, "TOP 10% most-confident predictions"), (99, "TOP 1% most-confident predictions")):
        thresh = np.percentile(conf, pct)
        simulate(conf >= thresh, name)

    print("=" * 92)
    print("Interpretation")
    print("=" * 92)
    print("  If gross P&L per trade is smaller than the round-trip cost — which is")
    print("  what a ~52% win rate on a ~12bps average move produces — then the")
    print("  statistical edge is real but economically unusable, and no amount of")
    print("  additional OHLCV feature engineering closes a gap that size.")

    print("\nDone.")


if __name__ == "__main__":
    main()
