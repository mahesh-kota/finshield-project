"""
FinShield - Market-only test on a verified SEBI case (Sadhna Broadcast).
Does the market-anomaly detector light up in SEBI's examination period
versus the quiet pre-period? Tests the market module only (the case was
YouTube-driven, so there is no Telegram data to test the social side).

  python src/sebi_market_test.py search Crystal Business System
  python src/sebi_market_test.py run TICKER.BO
"""
import sys
import pandas as pd
import yfinance as yf

PERIODS = {
    "pre-period":  ("2022-01-01", "2022-04-26"),
    "patch 1":     ("2022-04-27", "2022-07-14"),
    "patch 2":     ("2022-07-15", "2022-09-30"),
    "after":       ("2022-10-01", "2022-12-31"),
}

if len(sys.argv) >= 3 and sys.argv[1] == "search":
    try:
        res = yf.Search(" ".join(sys.argv[2:]), max_results=10)
        for q in res.quotes:
            print(q.get("symbol"), "|", q.get("shortname"), "|", q.get("exchange"))
    except Exception as e:
        print("Search failed:", e)
    sys.exit()

tkr = sys.argv[2]
df = pd.DataFrame()
for attempt in range(3):
    df = yf.download(tkr, start="2021-09-01", end="2023-01-31",
                     progress=False, timeout=30)
    if not df.empty:
        break
if df.empty:
    sys.exit(f"No data for {tkr}. Try another ticker, or this stock is not on yfinance.")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.reset_index()

df["ret"] = df["Close"].pct_change()
df["abn"] = df["ret"] - df["ret"].rolling(20).mean()
df["vol_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
df["anomaly"] = (df["vol_z"] > 2) & (df["abn"].abs() > 0.02)

rows = []
for name, (a, b) in PERIODS.items():
    s = df[(df["Date"] >= a) & (df["Date"] <= b)].dropna(subset=["vol_z", "abn"])
    rows.append([name, len(s), int(s["anomaly"].sum()),
                 round(s["anomaly"].mean(), 3) if len(s) else None,
                 round(s["vol_z"].mean(), 2) if len(s) else None,
                 round(s["abn"].abs().mean(), 4) if len(s) else None])
out = pd.DataFrame(rows, columns=["period", "days", "anomaly_days",
                                  "anomaly_share", "mean_vol_z", "mean_abs_abn_ret"])
print(out.to_string(index=False))
out.to_csv("docs/sebi_market_test_sadhna.csv", index=False)