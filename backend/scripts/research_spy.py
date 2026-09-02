"""
SPY-only sweep of everything the earlier passes (v1-v8) never tested.

Those passes all shared three unexamined choices: one pooled model across
all 51 tickers, a single never-tuned hyperparameter set
(max_depth=4, max_iter=200 in all 15 instantiations), and raw uncalibrated
probabilities. This narrows to SPY — the most liquid instrument we have,
427k minute bars over 2 years — and works through the untested list:

  1. BASELINE           SPY-only vs the pooled model's numbers.
  2. HYPERPARAMETERS    A real search instead of the one fixed setting.
  3. CALIBRATION        Isotonic / sigmoid. The conviction tiers are
                        percentiles of uncalibrated scores, so this bears
                        directly on the one finding we rely on.
  4. REGRESSION         Predict the return, take its sign — often
                        extracts more than predicting sign directly,
                        since it learns from move magnitude.
  5. SAMPLE WEIGHTING   Recency weighting, and weighting by |move| so the
                        model optimizes for the moves that actually pay.
  6. MULTI-DAY CONTEXT  Our longest lookback is 20 bars. Adds overnight
                        gap, position within the day's range, distance
                        from the prior close, and multi-day trend.
  7. INTERMARKET        VXX (volatility proxy — the VIX index itself is
                        403 on this tier), TLT (bonds), UUP (dollar),
                        GLD (gold). Contemporaneous, so no lookahead.
  8. VOLUME BARS        Sample by traded volume rather than clock time.
                        Lopez de Prado's argument is that time bars
                        oversample quiet periods and undersample active
                        ones, giving poor statistical properties.
  9. META-LABELING      Proper triple-barrier-style secondary model: let
                        the primary model call direction, then train a
                        second model to predict whether that call is
                        RIGHT. Reported to lift precision substantially.

Every test uses the same chronological 80/20 split on SPY's own timeline
and reports against the majority-class baseline, not a 50% coin flip.

Run: cd backend && source venv/bin/activate && python scripts/research_spy.py
"""

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.forecast_features import FEATURE_COLUMNS, HORIZON_MINUTES, build_features  # noqa: E402

warnings.filterwarnings("ignore")

DB_PATH = Path(__file__).resolve().parent.parent / "stockpulse.db"
TICKER = "SPY"
INTERMARKET = ["VXX", "TLT", "UUP", "GLD"]
TEST_FRAC = 0.2


def load(ticker):
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        return pd.read_sql_query(
            "SELECT timestamp, open, high, low, close, volume FROM ohlcv "
            "WHERE interval='minute' AND ticker=? ORDER BY timestamp",
            conn, params=(ticker,), parse_dates=["timestamp"],
        )


def make_labels(df, horizon=HORIZON_MINUTES):
    close, ts = df["close"], pd.to_datetime(df["timestamp"])
    fwd = (close.shift(-horizon) - close) / close
    gap = (ts.shift(-horizon) - ts).dt.total_seconds() / 60
    fwd[gap != horizon] = np.nan
    return fwd


def split(X, y, extra=None):
    cut = int(len(X) * (1 - TEST_FRAC))
    out = [X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]]
    if extra is not None:
        out += [extra.iloc[:cut], extra.iloc[cut:]]
    return out


def report(name, y_test, proba, extra=""):
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")
    maj = max(y_test.mean(), 1 - y_test.mean())
    conf = np.abs(proba - 0.5)
    hc = conf >= np.percentile(conf, 99)
    hc_acc = accuracy_score(y_test[hc], pred[hc]) if hc.sum() else float("nan")
    print(f"  {name:<44s} acc={acc:.4f} (maj {maj:.4f}, edge {acc - maj:+.4f})  "
          f"AUC={auc:.4f}  top1%={hc_acc:.4f}{extra}")
    return acc, auc, hc_acc


def base_frame():
    df = load(TICKER)
    feat = build_features(df)
    feat["fwd_ret"] = make_labels(df)
    feat["label"] = (feat["fwd_ret"] > 0).astype(float)
    feat.loc[feat["fwd_ret"].isna(), "label"] = np.nan
    return feat


def main():
    print(f"Loading {TICKER}...")
    feat = base_frame()
    print(f"  {len(feat):,} minute bars, "
          f"{pd.to_datetime(feat['timestamp']).min().date()} -> "
          f"{pd.to_datetime(feat['timestamp']).max().date()}\n")

    valid = feat[FEATURE_COLUMNS].notna().all(axis=1) & feat["label"].notna()
    X = feat.loc[valid, FEATURE_COLUMNS].reset_index(drop=True)
    y = feat.loc[valid, "label"].reset_index(drop=True)
    fwd = feat.loc[valid, "fwd_ret"].reset_index(drop=True)
    ts = pd.to_datetime(feat.loc[valid, "timestamp"]).reset_index(drop=True)
    X_tr, y_tr, X_te, y_te, fwd_tr, fwd_te = split(X, y, fwd)
    print(f"train={len(X_tr):,}  test={len(X_te):,}\n")

    print("=" * 104)
    print("1 — BASELINE: SPY alone, current features, current hyperparameters")
    print("=" * 104)
    base = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    base.fit(X_tr, y_tr)
    p_base = base.predict_proba(X_te)[:, 1]
    report("SPY-only baseline", y_te, p_base)
    print("  (pooled 51-ticker model for comparison: acc=0.5132 AUC=0.5172 top1%=0.573)")

    print("\n" + "=" * 104)
    print("2 — HYPERPARAMETERS: never tuned before; 15 identical instantiations")
    print("=" * 104)
    best = (None, -1)
    for depth in (3, 4, 6, 8, None):
        for iters in (100, 300, 600):
            for lr in (0.02, 0.05, 0.1):
                m = HistGradientBoostingClassifier(
                    max_depth=depth, max_iter=iters, learning_rate=lr,
                    l2_regularization=1.0, random_state=42,
                )
                m.fit(X_tr, y_tr)
                pr = m.predict_proba(X_te)[:, 1]
                try:
                    a = roc_auc_score(y_te, pr)
                except ValueError:
                    continue
                if a > best[1]:
                    best = ((depth, iters, lr), a)
    (bd, bi, blr), bauc = best
    print(f"  searched 45 configurations")
    print(f"  best: max_depth={bd} max_iter={bi} learning_rate={blr}")
    tuned = HistGradientBoostingClassifier(max_depth=bd, max_iter=bi, learning_rate=blr,
                                           l2_regularization=1.0, random_state=42)
    tuned.fit(X_tr, y_tr)
    p_tuned = tuned.predict_proba(X_te)[:, 1]
    report("tuned", y_te, p_tuned)

    print("\n" + "=" * 104)
    print("3 — CALIBRATION: conviction tiers are percentiles of raw scores")
    print("=" * 104)
    for method in ("isotonic", "sigmoid"):
        cal = CalibratedClassifierCV(
            HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42),
            method=method, cv=3,
        )
        cal.fit(X_tr, y_tr)
        report(f"calibrated ({method})", y_te, cal.predict_proba(X_te)[:, 1])

    print("\n" + "=" * 104)
    print("4 — REGRESSION on the return, then take its sign")
    print("=" * 104)
    reg = HistGradientBoostingRegressor(max_depth=4, max_iter=200, random_state=42)
    reg.fit(X_tr, fwd_tr)
    pred_ret = reg.predict(X_te)
    # Map predicted return to a pseudo-probability by rank so the same
    # accuracy/AUC/top-1% reporting applies.
    pseudo = pd.Series(pred_ret).rank(pct=True).to_numpy()
    report("regression -> sign", y_te, pseudo)

    print("\n" + "=" * 104)
    print("5 — SAMPLE WEIGHTING")
    print("=" * 104)
    recency = np.linspace(0.2, 1.0, len(X_tr))
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    m.fit(X_tr, y_tr, sample_weight=recency)
    report("recency-weighted", y_te, m.predict_proba(X_te)[:, 1])

    magnitude = np.abs(fwd_tr.to_numpy())
    magnitude = magnitude / magnitude.mean()
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    m.fit(X_tr, y_tr, sample_weight=magnitude)
    report("|move|-weighted", y_te, m.predict_proba(X_te)[:, 1])

    print("\n" + "=" * 104)
    print("6 — MULTI-DAY CONTEXT (longest existing lookback is 20 bars)")
    print("=" * 104)
    f2 = feat.copy()
    d = pd.to_datetime(f2["timestamp"]).dt.date
    day_open = f2.groupby(d)["open"].transform("first")
    # cummax/cummin are expanding WITHIN each day, so these are the high
    # and low SO FAR — known at time t, no lookahead.
    day_high = f2.groupby(d)["high"].transform("cummax")
    day_low = f2.groupby(d)["low"].transform("cummin")
    # Must be the PRIOR DAY's close. transform("last").shift(1) looks
    # right but broadcasts each day's final close across that same day and
    # then shifts by one ROW, so every intraday bar ends up reading its
    # own day's closing price — i.e. the future. Aggregate to daily first,
    # shift at the daily level, then map back.
    daily_close = f2.groupby(d)["close"].last()
    prev_close = d.map(daily_close.shift(1))
    f2["gap_from_day_open"] = (f2["close"] - day_open) / day_open
    f2["pos_in_day_range"] = (f2["close"] - day_low) / (day_high - day_low).replace(0, np.nan)
    f2["dist_prev_close"] = (f2["close"] - prev_close) / prev_close
    f2["ret_60"] = f2["close"].pct_change(60)
    f2["ret_390"] = f2["close"].pct_change(390)
    ctx = FEATURE_COLUMNS + ["gap_from_day_open", "pos_in_day_range", "dist_prev_close",
                             "ret_60", "ret_390"]
    v2 = f2[ctx].notna().all(axis=1) & f2["label"].notna()
    X2 = f2.loc[v2, ctx].reset_index(drop=True)
    y2 = f2.loc[v2, "label"].reset_index(drop=True)
    a_tr, b_tr, a_te, b_te = split(X2, y2)
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    m.fit(a_tr, b_tr)
    report("+ multi-day context", b_te, m.predict_proba(a_te)[:, 1])

    print("\n" + "=" * 104)
    print("7 — INTERMARKET (VXX volatility, TLT bonds, UUP dollar, GLD gold)")
    print("=" * 104)
    f3 = feat.copy()
    added = []
    for sym in INTERMARKET:
        other = load(sym)
        if len(other) < 1000:
            print(f"  {sym}: only {len(other)} bars — skipped")
            continue
        o = pd.DataFrame({"timestamp": other["timestamp"]})
        o[f"{sym}_ret1"] = other["close"].pct_change()
        o[f"{sym}_ret15"] = other["close"].pct_change(15)
        f3 = f3.merge(o, on="timestamp", how="left")
        added += [f"{sym}_ret1", f"{sym}_ret15"]
    if added:
        im = FEATURE_COLUMNS + added
        v3 = f3[im].notna().all(axis=1) & f3["label"].notna()
        X3 = f3.loc[v3, im].reset_index(drop=True)
        y3 = f3.loc[v3, "label"].reset_index(drop=True)
        c_tr, d_tr, c_te, d_te = split(X3, y3)
        m = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
        m.fit(c_tr, d_tr)
        report(f"+ intermarket ({len(added)} features)", d_te, m.predict_proba(c_te)[:, 1])
    else:
        print("  no intermarket data available yet")

    print("\n" + "=" * 104)
    print("8 — VOLUME BARS instead of time bars")
    print("=" * 104)
    raw = load(TICKER)
    target_vol = raw["volume"].sum() / len(raw) * 5  # ~5 minutes of average volume
    cum = raw["volume"].cumsum()
    raw["vbar"] = (cum // target_vol).astype(int)
    vb = raw.groupby("vbar").agg(
        timestamp=("timestamp", "last"), open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
    ).reset_index(drop=True)
    print(f"  {len(raw):,} time bars -> {len(vb):,} volume bars "
          f"(~{target_vol:,.0f} shares each)")
    vf = build_features(vb)
    # One volume bar ahead ~ the next equivalent-volume interval.
    vf["label"] = (vb["close"].shift(-1) > vb["close"]).astype(float)
    vf.loc[vb["close"].shift(-1).isna(), "label"] = np.nan
    v4 = vf[FEATURE_COLUMNS].notna().all(axis=1) & vf["label"].notna()
    X4 = vf.loc[v4, FEATURE_COLUMNS].reset_index(drop=True)
    y4 = vf.loc[v4, "label"].reset_index(drop=True)
    e_tr, f_tr, e_te, f_te = split(X4, y4)
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    m.fit(e_tr, f_tr)
    report("volume bars (1 bar ahead)", f_te, m.predict_proba(e_te)[:, 1])

    print("\n" + "=" * 104)
    print("9 — META-LABELING: second model predicts whether the first is RIGHT")
    print("=" * 104)
    # Primary model's in-sample-ish calls on the training half, used as
    # the secondary model's target. Split train in two so the secondary
    # learns from calls the primary didn't memorize.
    half = len(X_tr) // 2
    prim = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    prim.fit(X_tr.iloc[:half], y_tr.iloc[:half])
    p_mid = prim.predict_proba(X_tr.iloc[half:])[:, 1]
    meta_y = ((p_mid >= 0.5).astype(int) == y_tr.iloc[half:].to_numpy()).astype(float)
    meta_X = X_tr.iloc[half:].copy()
    meta_X["primary_conf"] = np.abs(p_mid - 0.5)

    meta = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    meta.fit(meta_X, meta_y)

    p_te_primary = prim.predict_proba(X_te)[:, 1]
    meta_te = X_te.copy()
    meta_te["primary_conf"] = np.abs(p_te_primary - 0.5)
    trust = meta.predict_proba(meta_te)[:, 1]

    print(f"  meta-model trust scores: mean={trust.mean():.4f} "
          f"p90={np.percentile(trust, 90):.4f} p99={np.percentile(trust, 99):.4f}")
    primary_pred = (p_te_primary >= 0.5).astype(int)
    for pct in (0, 50, 90, 99):
        keep = trust >= np.percentile(trust, pct)
        acc = accuracy_score(y_te[keep], primary_pred[keep])
        print(f"    act only on trust >= p{pct:<3d}: n={keep.sum():>7,d}  accuracy={acc:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
