"""
FinShield - Index-Level Event Study
For posts that mention NIFTY / BANKNIFTY, check whether the index
had an anomaly on the days around the post, versus its base rate.
Negative lag = anomaly BEFORE the post (posts reacting to market).
Positive lag = anomaly AFTER the post (posts leading the market).
"""
import pandas as pd
import numpy as np
from scipy.stats import binomtest

idx = pd.read_csv("data/index_features.csv", parse_dates=["Date"])
posts = pd.read_csv("data/telegram_posts_labelled.csv")
posts["day"] = (pd.to_datetime(posts["date"], format="mixed", utc=True)
                .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize())
txt = posts["text"].astype(str).str.upper()
mentions = {
    "NIFTY": txt.str.contains(r"\bNIFTY\b", regex=True),
    "BANKNIFTY": txt.str.contains(r"BANKNIFTY|\bBNF\b|BANK NIFTY", regex=True),
}
groups = {"all index posts": None, "Suspicious only": 2}

rows = []
for name, mask in mentions.items():
    ix = idx[idx["index"] == name].sort_values("Date").reset_index(drop=True)
    flag = ix["is_index_anomaly"].astype(bool).values
    base = flag.mean()
    dates = ix["Date"].values.astype("datetime64[ns]")
    for gname, lab in groups.items():
        sel = posts[mask] if lab is None else posts[mask & (posts["label"] == lab)]
        pos = np.searchsorted(dates, sel["day"].values.astype("datetime64[ns]"))
        pos = np.unique(pos[pos < len(ix)])
        for lag in range(-2, 3):
            p2 = pos + lag
            p2 = p2[(p2 >= 0) & (p2 < len(ix))]
            if len(p2) == 0:
                continue
            k = int(flag[p2].sum())
            pval = binomtest(k, len(p2), base, alternative="greater").pvalue
            rows.append([name, gname, len(pos), lag,
                         round(k / len(p2), 3), round(base, 3), round(pval, 3)])

res = pd.DataFrame(rows, columns=["index", "group", "event_days", "lag",
                                  "anomaly_share", "base_rate", "p_greater"])
print(res.to_string(index=False))
res.to_csv("docs/index_event_study.csv", index=False)