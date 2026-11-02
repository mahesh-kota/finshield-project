"""
FinShield - Replication check: does the NIFTY/BANKNIFTY lag +1/+2 hint
appear separately in each channel (non-overlapping periods)?
Base rate is computed over each channel's own date span.
"""
import pandas as pd
import numpy as np
from scipy.stats import binomtest

idx = pd.read_csv("data/index_features.csv", parse_dates=["Date"])
posts = pd.read_csv("data/history_predictions.csv")
posts["day"] = (pd.to_datetime(posts["date"], format="mixed", utc=True)
                .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize())
txt = posts["text"].astype(str).str.upper()

patterns = {"NIFTY": r"\bNIFTY\b", "BANKNIFTY": r"BANKNIFTY|\bBNF\b|BANK NIFTY"}
splits = {
    "best_intraday_stock": posts["channel"] == "best_intraday_stock",
    "Nifty451": posts["channel"] == "Nifty451",
}

rows = []
for name, pat in patterns.items():
    ix = idx[idx["index"] == name].sort_values("Date").reset_index(drop=True)
    flag = ix["is_index_anomaly"].astype(bool).values
    dates = ix["Date"].values.astype("datetime64[ns]")
    has = txt.str.contains(pat, regex=True)
    for sname, smask in splits.items():
        sel = posts[has & smask]
        if sel.empty:
            continue
        lo, hi = sel["day"].min(), sel["day"].max()
        in_period = ((ix["Date"] >= lo) & (ix["Date"] <= hi)).values
        base = flag[in_period].mean()
        pos = np.searchsorted(dates, sel["day"].values.astype("datetime64[ns]"))
        pos = np.unique(pos[pos < len(ix)])
        for lag in (1, 2):
            p2 = pos + lag
            p2 = p2[p2 < len(ix)]
            k = int(flag[p2].sum())
            pv = binomtest(k, len(p2), base, alternative="greater").pvalue
            rows.append([name, sname, len(pos), lag,
                         round(k / len(p2), 3), round(base, 3), round(pv, 4)])

res = pd.DataFrame(rows, columns=["index", "channel", "event_days", "lag",
                                  "anomaly_share", "base_rate", "p_greater"])
print(res.to_string(index=False))