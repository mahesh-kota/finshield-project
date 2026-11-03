"""
FinShield - Market anomaly detector v3.
Fixes trailing-baseline contamination by using:
  (a) a fixed 2021 baseline (pre-event), and
  (b) a within-window volume percentile rank.
"""
import pandas as pd
import yfinance as yf

tkr = "CRYSTAL.BO"
df = yf.download(tkr, start="2021-01-01", end="2023-01-31", progress=False, timeout=30)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.reset_index()

base_2021 = df[(df["Date"] >= "2021-01-01") & (df["Date"] <= "2021-12-31")]["Volume"].median()
print(f"2021 median volume (fixed baseline): {base_2021:,.0f}\n")

df["vol_ratio_fixed"] = df["Volume"] / base_2021
df["vol_rank_pct"] = df["Volume"].rank(pct=True)
df["top5pct"] = df["vol_rank_pct"] >= 0.95

P = {"pre-period": ("2022-01-01", "2022-04-26"),
     "patch 1":    ("2022-04-27", "2022-07-14"),
     "patch 2":    ("2022-07-15", "2022-09-30"),
     "after":      ("2022-10-01", "2022-12-31")}

rows = []
for k, (a, b) in P.items():
    s = df[(df["Date"] >= a) & (df["Date"] <= b)].dropna(subset=["vol_ratio_fixed"])
    n = len(s)
    rows.append([k, n,
                 round(s["vol_ratio_fixed"].median(), 1) if n else None,
                 round((s["vol_ratio_fixed"] > 3).mean(), 3) if n else None,
                 round((s["vol_ratio_fixed"] > 10).mean(), 3) if n else None,
                 round(s["top5pct"].mean(), 3) if n else None])

out = pd.DataFrame(rows, columns=["period", "days", "median_ratio_vs_2021",
                                  "share_gt3x", "share_gt10x", "share_top5pct"])
print(out.to_string(index=False))
out.to_csv("docs/sebi_market_test_sadhna_v3.csv", index=False)
