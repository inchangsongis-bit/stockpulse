"""
Error analysis: what is different about the moments the model gets
confidently WRONG versus confidently RIGHT?

Motivation: if the model's confident mistakes cluster around identifiable
market conditions — a market-wide shock, an unusual volume surge, the
opening minutes — then either those conditions can become features, or
the model can simply decline to call those moments.

Historical news turned out not to be usable for this (see the notes in
scripts/fetch_market_state.py and the summary below): Finnhub's free tier
carries roughly 12 months of per-ticker company news, not the ~2 years
our price history now spans, and its general/macro market feed is
live-only with no history at all. The macro and global-news series the
question really calls for simply isn't retrievable.

So this measures the OBSERVABLE FOOTPRINT of news instead, which is
available at minute resolution for the entire history, costs nothing, and
carries no lookahead risk because it's all contemporaneous price and
volume:

    mkt_dispersion  cross-sectional std of 1-min returns across the
                    watchlist. Low = everything moving together (a
                    market-wide event); high = idiosyncratic, stock-
                    specific moves.
    mkt_abs_move    cross-sectional mean |1-min return| — overall market
                    activity level right now.
    mkt_vol_surge   cross-sectional mean volume ratio — is the whole
                    market unusually busy this minute?
    mkt_breadth     fraction of the watchlist up this minute — how
                    one-sided the tape is.

A note on why news itself is treacherous here even where it IS available:
most financial headlines are REACTIVE. A story timestamped 10:32 saying
"shares slide on guidance" describes a move that already happened at
10:30. Feeding "was there news in the last N minutes" to a model that
predicts forward returns invites it to learn the news's description of
the past, and many aggregators stamp ingestion time rather than
publication time, so the "news" can even postdate the move being
predicted. The market-state features above sidestep this entirely: they
are facts about the tape at time t, used to predict t+5.

Run: cd backend && source venv/bin/activate && python scripts/research_forecast_v8.py
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.forecast_features import FEATURE_COLUMNS, HORIZON_MINUTES, build_features, build_labels  # noqa: E402

RTH_START_MIN, RTH_END_MIN = 390, 780
MARKET_FEATURES = ["mkt_dispersion", "mkt_abs_move", "mkt_vol_surge", "mkt_breadth"]
DB_PATH = Path(__file__).resolve().parent.parent / "stockpulse.db"


def load_minute_bars_by_ticker() -> dict:
    """
    Read straight into a DataFrame via sqlite3 rather than materializing
    ~17M ORM rows as Python dicts — at this dataset size the dict-of-lists
    approach costs several GB of RAM and minutes of overhead.
    """
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        df = pd.read_sql_query(
            "SELECT ticker, timestamp, open, high, low, close, volume "
            "FROM ohlcv WHERE interval = 'minute' ORDER BY ticker, timestamp",
            conn,
            parse_dates=["timestamp"],
        )
    return {t: g.reset_index(drop=True) for t, g in df.groupby("ticker", sort=False)}


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    mod = ts.dt.hour * 60 + ts.dt.minute
    return df[(mod >= RTH_START_MIN) & (mod <= RTH_END_MIN)].reset_index(drop=True)


def build_market_state(bars_by_ticker: dict) -> pd.DataFrame:
    """Cross-sectional market-state series indexed by timestamp — the
    observable footprint of whatever news is moving the tape."""
    frames = []
    for ticker, bars in bars_by_ticker.items():
        d = pd.DataFrame({"timestamp": bars["timestamp"]})
        d["r1"] = bars["close"].pct_change()
        vol_sma = bars["volume"].rolling(20, min_periods=20).mean()
        d["vr"] = bars["volume"] / vol_sma.replace(0, np.nan)
        frames.append(d)
    stacked = pd.concat(frames, ignore_index=True).dropna(subset=["r1"])

    # Derive the per-row quantities first so every aggregate below is a
    # vectorized groupby column op. GroupBy.apply with a lambda would be
    # orders of magnitude slower across ~200k distinct timestamps.
    stacked["abs_r1"] = stacked["r1"].abs()
    stacked["is_up"] = (stacked["r1"] > 0).astype(float)

    g = stacked.groupby("timestamp")
    out = pd.DataFrame({
        "mkt_dispersion": g["r1"].std(),
        "mkt_abs_move": g["abs_r1"].mean(),
        "mkt_vol_surge": g["vr"].mean(),
        "mkt_breadth": g["is_up"].mean(),
    }).reset_index()
    return out


def assemble(bars_by_ticker, market_state=None, features=None):
    features = features or FEATURE_COLUMNS
    tr_X, tr_y = [], []
    te_X, te_y, te_meta = [], [], []
    for ticker, bars in bars_by_ticker.items():
        if len(bars) < 200:
            continue
        featured = build_features(bars)
        labels = build_labels(bars)

        close = bars["close"]
        ts = pd.to_datetime(bars["timestamp"])
        fwd = (close.shift(-HORIZON_MINUTES) - close) / close

        if market_state is not None:
            featured = featured.merge(market_state, on="timestamp", how="left")

        valid = featured[features].notna().all(axis=1) & labels.notna()
        X = featured.loc[valid, features].reset_index(drop=True)
        y = labels.loc[valid].reset_index(drop=True)
        if len(X) < 100:
            continue

        meta = pd.DataFrame({
            "ticker": ticker,
            "timestamp": ts.loc[valid].reset_index(drop=True),
            "fwd_bps": fwd.loc[valid].reset_index(drop=True) * 10000,
        })
        for col in (market_state.columns if market_state is not None else []):
            if col != "timestamp":
                meta[col] = featured.loc[valid, col].reset_index(drop=True)

        cut = int(len(X) * 0.8)
        tr_X.append(X.iloc[:cut]); tr_y.append(y.iloc[:cut])
        te_X.append(X.iloc[cut:]); te_y.append(y.iloc[cut:]); te_meta.append(meta.iloc[cut:])

    return (
        pd.concat(tr_X, ignore_index=True), pd.concat(tr_y, ignore_index=True),
        pd.concat(te_X, ignore_index=True), pd.concat(te_y, ignore_index=True),
        pd.concat(te_meta, ignore_index=True),
    )


def main():
    print("Loading minute bars (regular trading hours only)...")
    bars = load_minute_bars_by_ticker()
    bars = {t: filter_rth(b) for t, b in bars.items()}
    bars = {t: b for t, b in bars.items() if len(b) > 500}
    total = sum(len(b) for b in bars.values())
    print(f"  {len(bars)} tickers, {total:,} RTH bars")

    print("Building cross-sectional market-state series...")
    mkt = build_market_state(bars)
    print(f"  {len(mkt):,} distinct minutes\n")

    X_train, y_train, X_test, y_test, meta = assemble(bars, market_state=mkt)
    print(f"Training on {len(X_train):,} rows, analysing {len(X_test):,} held-out predictions\n")

    model = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    conf = np.abs(proba - 0.5)
    pred = (proba >= 0.5).astype(int)
    correct = (pred == y_test.to_numpy()).astype(bool)
    high_conf = conf >= np.percentile(conf, 90)

    meta = meta.copy()
    meta["conf"] = conf
    meta["correct"] = correct
    meta["high_conf"] = high_conf
    meta["minute_of_day"] = pd.to_datetime(meta["timestamp"]).dt.hour * 60 + pd.to_datetime(meta["timestamp"]).dt.minute
    meta["abs_move_bps"] = meta["fwd_bps"].abs()

    print("=" * 104)
    print("W — Profile of the four outcome groups (top-decile confidence vs the rest)")
    print("=" * 104)
    groups = {
        "confident + RIGHT": meta[meta.high_conf & meta.correct],
        "confident + WRONG": meta[meta.high_conf & ~meta.correct],
        "unsure    + RIGHT": meta[~meta.high_conf & meta.correct],
        "unsure    + WRONG": meta[~meta.high_conf & ~meta.correct],
    }
    cols = ["abs_move_bps", "mkt_dispersion", "mkt_abs_move", "mkt_vol_surge", "mkt_breadth"]
    print(f"  {'group':<20s}{'n':>9s}" + "".join(f"{c.replace('mkt_',''):>16s}" for c in cols))
    for name, g in groups.items():
        vals = "".join(f"{g[c].mean():>16.5f}" for c in cols)
        print(f"  {name:<20s}{len(g):>9,d}{vals}")

    print("\n  Read: if 'confident + WRONG' shows systematically different market state than")
    print("  'confident + RIGHT', those conditions are a filter the model could use.")

    print("\n" + "=" * 104)
    print("X — Accuracy by market condition (high-confidence predictions only)")
    print("=" * 104)
    hc = meta[meta.high_conf]
    for col in ["mkt_dispersion", "mkt_vol_surge", "mkt_abs_move", "mkt_breadth"]:
        try:
            q = pd.qcut(hc[col], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        print(f"\n  by {col}:")
        for b in sorted(pd.unique(q.dropna())):
            sub = hc[q == b]
            print(f"    quintile {int(b) + 1}: n={len(sub):>7,d}  accuracy={sub['correct'].mean():.4f}  "
                  f"avg |move|={sub['abs_move_bps'].mean():>6.2f} bps")

    print("\n" + "=" * 104)
    print("Y — Accuracy by time of day (high-confidence predictions only)")
    print("=" * 104)
    hc_by_hour = hc.groupby(hc["minute_of_day"] // 30 * 30)
    print(f"  {'window (PT)':<16s}{'n':>9s}{'accuracy':>11s}{'avg |move|':>13s}")
    for start, g in hc_by_hour:
        h, m = divmod(int(start), 60)
        print(f"  {f'{h:02d}:{m:02d}':<16s}{len(g):>9,d}{g['correct'].mean():>11.4f}{g['abs_move_bps'].mean():>12.2f}b")

    print("\n" + "=" * 104)
    print("Z — Do the market-state features actually improve the model?")
    print("=" * 104)
    base_acc = accuracy_score(y_test, pred)
    base_auc = roc_auc_score(y_test, proba)
    majority = max(y_test.mean(), 1 - y_test.mean())
    print(f"  base features only            : acc={base_acc:.4f}  AUC={base_auc:.4f}  "
          f"(majority {majority:.4f})")

    aug_features = list(FEATURE_COLUMNS) + MARKET_FEATURES
    aXtr, aytr, aXte, ayte, _ = assemble(bars, market_state=mkt, features=aug_features)
    am = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    am.fit(aXtr, aytr)
    ap = am.predict_proba(aXte)[:, 1]
    a_acc = accuracy_score(ayte, (ap >= 0.5).astype(int))
    a_auc = roc_auc_score(ayte, ap)
    print(f"  + market-state features       : acc={a_acc:.4f}  AUC={a_auc:.4f}  "
          f"(delta {a_acc - base_acc:+.4f} / {a_auc - base_auc:+.4f})")

    aconf = np.abs(ap - 0.5)
    ahc = aconf >= np.percentile(aconf, 99)
    a_hc_acc = accuracy_score(ayte[ahc], (ap[ahc] >= 0.5).astype(int))
    hc99 = conf >= np.percentile(conf, 99)
    b_hc_acc = accuracy_score(y_test[hc99], pred[hc99])
    print(f"\n  top-1% conviction accuracy    : base {b_hc_acc:.4f}  ->  with market state {a_hc_acc:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
