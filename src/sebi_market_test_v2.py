import pandas as pd
import yfinance as yf

tkr = "CRYSTAL.BO"
df = yf.download(tkr, start="2021-09-01", end="2023-01-31", progress=False, timeout=30)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.reset_index()

df["ret"] = df["Close"].pct_change()

df["vol_z_20"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
df["abn"] = df["ret"] - df["ret"].rolling(20).mean()
df["anomaly_v1"] = (df["vol_z_20"] > 2) & (df["abn"].abs() > 0.02)

df["vol_med_120"] = df["Volume"].rolling(120, min_periods=60).median()
df["vol_ratio"] = df["Volume"] / df["vol_med_120"]

P = {"pre-period": ("2022-01-01", "2022-04-26"),
     "patch 1":    ("2022-04-27", "2022-07-14"),
     "patch 2":    ("2022-07-15", "2022-09-30"),
     "after":      ("2022-10-01", "2022-12-31")}

rows = []
for k, (a, b) in P.items():
    s = df[(df["Date"] >= a) & (df["Date"] <= b)].dropna(subset=["vol_ratio"])
    n = len(s)
    rows.append([k, n,
                 round(s["anomaly_v1"].mean(), 3) if n else None,
                 round((s["vol_ratio"] > 3).mean(), 3) if n else None,
                 round((s["vol_ratio"] > 5).mean(), 3) if n else None,
                 round((s["vol_ratio"] > 10).mean(), 3) if n else None,
                 round(s["vol_ratio"].median(), 2) if n else None])

out = pd.DataFrame(rows, columns=["period", "days", "v1_share",
                                  "v2_3x", "v2_5x", "v2_10x", "median_vol_ratio"])
print(out.to_string(index=False))
out.to_csv("docs/sebi_market_test_sadhna_v2.csv", index=False)
