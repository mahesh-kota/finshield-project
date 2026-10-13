"""
FinShield - event study: do MuRIL-flagged suspicious posts precede market stress?

Method
  * Event days  : trading days with >=1 post predicted Suspicious (muril_pred == 2).
                  A post is mapped to the first trading day on/after its date.
  * Outcomes    : market-wide daily series (mean over the 47 stocks) of
                    mkt_abs_ar   = mean |abnormal_return|
                    mkt_vol_z    = mean volume_zscore
                    mkt_anom     = share of stocks flagged is_anomaly
  * Lags        : h = -2..+2 trading days around the event day.
                  h < 0 is a PLACEBO: elevated stress BEFORE posts would mean the
                  posts react to the market rather than lead it.
  * Test        : permutation. Draw the same number of random non-event trading
                  days 10,000 times; p = share of draws with mean >= observed.
  * Optional    : ticker-level version if enough posts have non-empty
                  tickers_mentioned (printed first so you can see the count).

Run from the repo root:  python event_study.py
"""
import ast
import os
import numpy as np
import pandas as pd

# ----------------------------- CONFIG ------------------------------------
MARKET_CSV = "data/market_features.csv"
POSTS_CSV = "data/features.csv"
SUSPICIOUS_CLASS = 2
LAGS = [-2, -1, 0, 1, 2]
N_PERM = 10_000
MIN_TICKER_EVENTS = 20
SEED = 42
RESULTS_CSV = "results/event_study.csv"
# --------------------------------------------------------------------------

rng = np.random.default_rng(SEED)


def load_market():
    m = pd.read_csv(MARKET_CSV, parse_dates=["Date"])
    m["abs_ar"] = m["abnormal_return"].abs()
    m["is_anomaly"] = m["is_anomaly"].astype(float)
    daily = m.groupby("Date").agg(
        mkt_abs_ar=("abs_ar", "mean"),
        mkt_vol_z=("volume_zscore", "mean"),
        mkt_anom=("is_anomaly", "mean"),
    ).sort_index()
    return m, daily


def load_posts(trading_days):
    p = pd.read_csv(POSTS_CSV)
    p["date"] = pd.to_datetime(p["date"], utc=True, format="mixed").dt.tz_localize(None).dt.normalize()
    pos = trading_days.searchsorted(p["date"].values, side="left")
    ok = pos < len(trading_days)
    p = p[ok].copy()
    p["trade_day"] = trading_days[pos[ok]]
    return p


def perm_test(series, event_days, all_days, h, n_events):
    """Mean of series at (day + h) over events vs random non-event days."""
    idx = {d: i for i, d in enumerate(all_days)}
    vals = series.reindex(all_days).values

    def at(days):
        out = []
        for d in days:
            j = idx[d] + h
            if 0 <= j < len(vals) and not np.isnan(vals[j]):
                out.append(vals[j])
        return np.array(out)

    obs_vals = at(event_days)
    if len(obs_vals) == 0:
        return None
    obs = obs_vals.mean()
    pool = [d for d in all_days if d not in set(event_days)]
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        draw = rng.choice(len(pool), size=len(event_days), replace=False)
        v = at([pool[i] for i in draw])
        null[b] = v.mean() if len(v) else np.nan
    null_mean = np.nanmean(null)
    p = float((np.nansum(null >= obs) + 1) / (np.sum(~np.isnan(null)) + 1))
    return {"obs_mean": obs, "null_mean": null_mean, "diff": obs - null_mean,
            "p_one_sided": p, "n_events": len(obs_vals)}


def market_wide(daily, posts):
    ev = posts[posts["muril_pred"] == SUSPICIOUS_CLASS]["trade_day"].drop_duplicates().tolist()
    all_days = list(daily.index)
    print(f"\nEvent days (>=1 predicted-Suspicious post): {len(ev)} of {len(all_days)} trading days")
    rows = []
    for name in ["mkt_abs_ar", "mkt_vol_z", "mkt_anom"]:
        for h in LAGS:
            r = perm_test(daily[name], ev, all_days, h, len(ev))
            if r:
                rows.append({"level": "market-wide", "outcome": name, "lag": h, **r})
    return rows


def ticker_level(m, posts, trading_days):
    parsed = posts["tickers_mentioned"].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else [])
    posts = posts.assign(tick=parsed)
    n_nonempty = int((posts["tick"].str.len() > 0).sum())
    print(f"\nPosts with non-empty tickers_mentioned: {n_nonempty} of {len(posts)}")

    sus = posts[posts["muril_pred"] == SUSPICIOUS_CLASS]
    pairs = set()
    for _, r in sus.iterrows():
        for t in r["tick"]:
            t = t if t.endswith(".NS") else f"{t}.NS"
            pairs.add((t, r["trade_day"]))
    print(f"Distinct (ticker, day) Suspicious events: {len(pairs)}")
    if len(pairs) < MIN_TICKER_EVENTS:
        print(f"Fewer than {MIN_TICKER_EVENTS} - skipping ticker-level test (too few to be meaningful).")
        return []

    wide = m.pivot(index="Date", columns="ticker", values="abs_ar").reindex(trading_days)
    pos_of = {d: i for i, d in enumerate(trading_days)}
    ev = [(t, d) for t, d in pairs if t in wide.columns and d in pos_of]
    valid = wide.stack().index.tolist()
    ev_set = set(ev)
    pool = [(t, d) for d, t in valid if (t, d) not in ev_set]
    rows = []
    for h in LAGS:
        def stat(pairs_):
            out = []
            for t, d in pairs_:
                j = pos_of[d] + h
                if 0 <= j < len(trading_days):
                    v = wide.iloc[j][t]
                    if not np.isnan(v):
                        out.append(v)
            return np.array(out)
        o = stat(ev)
        if len(o) == 0:
            continue
        null = np.empty(N_PERM)
        for b in range(N_PERM):
            draw = rng.choice(len(pool), size=len(ev), replace=False)
            v = stat([pool[i] for i in draw])
            null[b] = v.mean() if len(v) else np.nan
        p = float((np.nansum(null >= o.mean()) + 1) / (np.sum(~np.isnan(null)) + 1))
        rows.append({"level": "ticker", "outcome": "abs_ar", "lag": h, "obs_mean": o.mean(),
                     "null_mean": np.nanmean(null), "diff": o.mean() - np.nanmean(null),
                     "p_one_sided": p, "n_events": len(o)})
    return rows


def main():
    m, daily = load_market()
    trading_days = daily.index
    posts = load_posts(trading_days)
    print(f"{len(posts)} posts mapped to trading days; "
          f"{(posts['muril_pred'] == SUSPICIOUS_CLASS).sum()} predicted Suspicious")

    rows = market_wide(daily, posts) + ticker_level(m, posts, trading_days)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print("\n=== Event study (lag 0 = event day; negative lags = placebo) ===")
    print(res.round(5).to_string(index=False))

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    res.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
