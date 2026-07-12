import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load stock universe
with open("data/stock_universe.txt") as f:
    tickers = [line.strip() for line in f if line.strip()]

print(f"Processing {len(tickers)} stocks...")

# Date range — 1 year of data for reliable z-score baselines
END_DATE = datetime.today().strftime("%Y-%m-%d")
START_DATE = "2022-01-01"
results = []

for ticker in tickers:
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if df.empty or len(df) < 30:
            print(f"SKIP {ticker} - insufficient data")
            continue

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Daily return
        df["daily_return"] = df["Close"].pct_change()

        # Abnormal return = daily return - rolling 20-day mean return
        df["mean_return"] = df["daily_return"].rolling(20).mean()
        df["abnormal_return"] = df["daily_return"] - df["mean_return"]

        # Volume z-score = (today volume - rolling mean) / rolling std
        df["vol_mean"] = df["Volume"].rolling(20).mean()
        df["vol_std"] = df["Volume"].rolling(20).std()
        df["volume_zscore"] = (df["Volume"] - df["vol_mean"]) / df["vol_std"]

        # Price range anomaly = (High - Low) / Close
        df["price_range_ratio"] = (df["High"] - df["Low"]) / df["Close"]
        df["range_zscore"] = (df["price_range_ratio"] - df["price_range_ratio"].rolling(20).mean()) / df["price_range_ratio"].rolling(20).std()

        # Flag anomaly days: volume z-score > 2 AND abnormal return > 2%
        df["is_anomaly"] = (df["volume_zscore"] > 2) & (df["abnormal_return"].abs() > 0.02)

        # Keep last 90 days for analysis
        df_recent = df.copy()
        df_recent["ticker"] = ticker
        df_recent = df_recent.reset_index()
        df_recent = df_recent[["ticker", "Date", "Close", "Volume",
                                "daily_return", "abnormal_return",
                                "volume_zscore", "range_zscore", "is_anomaly"]]
        results.append(df_recent)

        anomaly_count = df_recent["is_anomaly"].sum()
        print(f"OK {ticker} - {len(df_recent)} days ({df_recent['Date'].min()} to {df_recent['Date'].max()}), {anomaly_count} anomaly days flagged")

    except Exception as e:
        print(f"ERROR {ticker}: {e}")

# Combine all stocks
if results:
    market_df = pd.concat(results, ignore_index=True)
    market_df.to_csv("data/market_features.csv", index=False)
    print(f"\nTotal rows: {len(market_df)}")
    print(f"Total anomaly days flagged: {market_df['is_anomaly'].sum()}")
    print(f"Stocks with at least 1 anomaly: {market_df.groupby('ticker')['is_anomaly'].any().sum()}")
    print(f"\nTop 5 most anomalous stocks:")
    print(market_df.groupby("ticker")["is_anomaly"].sum().sort_values(ascending=False).head())