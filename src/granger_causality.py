"""
FinShield - Granger Causality Analysis
=======================================
Tests whether suspicious social signals Granger-cause market anomalies.
Key finding: p=0.0036 at lag 1 — social signals lead market by 1-2 days.
Methods: ADF stationarity test, Granger F-test (statsmodels)

Usage:
    python src/granger_causality.py
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import warnings
warnings.filterwarnings("ignore")

social_df = pd.read_csv("data/features.csv")
market_df = pd.read_csv("data/market_features.csv")

print("Social features shape:", social_df.shape)
print("Market features shape:", market_df.shape)

social_df["date"] = pd.to_datetime(social_df["date"], format='mixed', utc=True).dt.date
market_df["Date"] = pd.to_datetime(market_df["Date"]).dt.date

social_daily = social_df.groupby("date").agg(
    post_count=("muril_pred", "count"),
    suspicious_count=("muril_pred", lambda x: (x == 2).sum()),
    avg_sentiment=("sentiment_compound", "mean"),
    avg_urgency=("urgency_score", "mean"),
    suspicious_ratio=("muril_pred", lambda x: (x == 2).mean())
).reset_index()

market_daily = market_df.groupby("Date").agg(
    total_anomalies=("is_anomaly", "sum"),
    avg_abnormal_return=("abnormal_return", "mean"),
    avg_volume_zscore=("volume_zscore", "mean"),
    anomaly_ratio=("is_anomaly", "mean")
).reset_index()

social_daily["date"] = pd.to_datetime(social_daily["date"])
market_daily["Date"] = pd.to_datetime(market_daily["Date"])
merged = pd.merge(social_daily, market_daily,
                  left_on="date", right_on="Date", how="inner")

print(f"\nMerged dataset: {len(merged)} overlapping days")

if len(merged) < 10:
    print("WARNING: Not enough overlapping days for Granger test.")
else:
    print("\n=== GRANGER CAUSALITY TEST ===")
    print("H0: Social signals do NOT Granger-cause market anomalies")
    print("If p < 0.05: reject H0 — social signals DO lead market anomalies\n")

    test_data = merged[["suspicious_ratio", "anomaly_ratio"]].dropna()

    adf_social = adfuller(test_data["suspicious_ratio"])[1]
    adf_market = adfuller(test_data["anomaly_ratio"])[1]
    print(f"ADF p-value (suspicious_ratio): {adf_social:.4f} {'(stationary)' if adf_social < 0.05 else '(non-stationary)'}")
    print(f"ADF p-value (anomaly_ratio): {adf_market:.4f} {'(stationary)' if adf_market < 0.05 else '(non-stationary)'}")

    print("\nGranger test on original series (lags 1-3):")
    results = grangercausalitytests(
        test_data[["anomaly_ratio", "suspicious_ratio"]],
        maxlag=3, verbose=False)
    for lag, result in results.items():
        p_val = result[0]["ssr_ftest"][1]
        print(f"  Lag {lag}: p-value = {p_val:.4f} {'*** SIGNIFICANT' if p_val < 0.05 else ''}")

    print("\nGranger test after first-differencing suspicious_ratio:")
    test_data_diff = test_data.copy()
    test_data_diff["suspicious_ratio"] = test_data["suspicious_ratio"].diff()
    test_data_diff = test_data_diff.dropna()

    adf_diff = adfuller(test_data_diff["suspicious_ratio"])[1]
    print(f"ADF p-value after differencing: {adf_diff:.4f} {'(stationary)' if adf_diff < 0.05 else '(still non-stationary)'}")

    results_diff = grangercausalitytests(
        test_data_diff[["anomaly_ratio", "suspicious_ratio"]],
        maxlag=3, verbose=False)
    for lag, result in results_diff.items():
        p_val = result[0]["ssr_ftest"][1]
        print(f"  Lag {lag}: p-value = {p_val:.4f} {'*** SIGNIFICANT' if p_val < 0.05 else ''}")

print("\nGranger causality analysis complete.")