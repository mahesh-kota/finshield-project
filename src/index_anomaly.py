"""
FinShield - Index-Level Market Features
Most posts discuss NIFTY/BANKNIFTY options, not single stocks.
This builds index anomaly features so posts can be aligned
with the instrument they actually talk about.
"""
import pandas as pd
import yfinance as yf

INDICES = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
frames = []

for name, tkr in INDICES.items():
    df = pd.DataFrame()
    for attempt in range(3):
        df = yf.download(tkr, start="2022-01-01", progress=False, timeout=30)
        if not df.empty:
            break
        print(f"{name}: attempt {attempt + 1} failed, retrying...")
    if df.empty:
        print(f"SKIP {name} - no data after 3 attempts")
        continue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["daily_return"] = df["Close"].pct_change()
    df["abnormal_return"] = df["daily_return"] - df["daily_return"].rolling(20).mean()
    df["abn_z"] = df["abnormal_return"] / df["daily_return"].rolling(20).std()
    rng = (df["High"] - df["Low"]) / df["Close"]
    df["range_zscore"] = (rng - rng.rolling(20).mean()) / rng.rolling(20).std()
    df["is_index_anomaly"] = (df["abn_z"].abs() > 2) | (df["range_zscore"] > 2)
    df["index"] = name
    frames.append(df[["index", "Date", "Close", "daily_return", "abnormal_return",
                      "abn_z", "range_zscore", "is_index_anomaly"]])

if frames:
    out = pd.concat(frames, ignore_index=True)
    out.to_csv("data/index_features.csv", index=False)
    print(out.groupby("index")["is_index_anomaly"].agg(["count", "sum"]))

posts = pd.read_csv("data/telegram_posts_labelled.csv")
txt = posts["text"].astype(str).str.upper()
patterns = {
    "NIFTY": r"\bNIFTY\b",
    "BANKNIFTY": r"BANKNIFTY|\bBNF\b|BANK NIFTY",
    "SENSEX": r"SENSEX",
}
print()
for name, pat in patterns.items():
    n = txt.str.contains(pat, regex=True).sum()
    print(f"{name}: mentioned in {n} of {len(posts)} posts")