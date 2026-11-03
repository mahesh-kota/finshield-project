import pandas as pd
import yfinance as yf

tkr = "CRYSTAL.BO"
df = yf.download(tkr, start="2021-09-01", end="2023-01-31", progress=False, timeout=30)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.reset_index()

P = {"pre-period": ("2022-01-01", "2022-04-26"),
     "patch 1": ("2022-04-27", "2022-07-14"),
     "patch 2": ("2022-07-15", "2022-09-30")}
order_avg = {"pre-period": 43740, "patch 1": 259561, "patch 2": 1892638}

base = None
for k, (a, b) in P.items():
    s = df[(df["Date"] >= a) & (df["Date"] <= b)]
    m = s["Volume"].mean()
    base = base or m
    print(f"{k:11} days={len(s):3} mean_vol={m:12,.0f} x_vs_pre={m/base:6.1f}  order_says={order_avg[k]:,}")

print()
w = df[(df["Date"] >= "2022-07-11") & (df["Date"] <= "2022-07-20")]
print(w[["Date", "Close", "Volume"]].to_string(index=False))