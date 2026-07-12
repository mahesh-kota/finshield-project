import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# Load both feature sets
social_df = pd.read_csv("data/features.csv")
market_df = pd.read_csv("data/market_features.csv")

# --- Social daily aggregation ---
social_df["date"] = pd.to_datetime(social_df["date"], utc=True).dt.date
social_daily = social_df.groupby("date").agg(
    post_count=("muril_pred", "count"),
    suspicious_ratio=("muril_pred", lambda x: (x == 2).mean()),
    avg_sentiment=("sentiment_compound", "mean"),
    avg_urgency=("urgency_score", "mean"),
    muril_confidence=("muril_confidence", "mean")
).reset_index()
social_daily["date"] = pd.to_datetime(social_daily["date"])

# --- Market daily aggregation ---
market_df["Date"] = pd.to_datetime(market_df["Date"])
market_daily = market_df.groupby("Date").agg(
    total_anomalies=("is_anomaly", "sum"),
    avg_abnormal_return=("abnormal_return", "mean"),
    avg_volume_zscore=("volume_zscore", "mean"),
    avg_range_zscore=("range_zscore", "mean"),
    anomaly_ratio=("is_anomaly", "mean")
).reset_index()

# --- Merge ---
merged = pd.merge(social_daily, market_daily,
                  left_on="date", right_on="Date", how="inner")
print(f"Merged dataset: {len(merged)} days")

# --- Feature matrix for Isolation Forest ---
feature_cols = [
    "suspicious_ratio",
    "avg_sentiment",
    "avg_urgency",
    "muril_confidence",
    "avg_abnormal_return",
    "avg_volume_zscore",
    "avg_range_zscore",
    "anomaly_ratio"
]

X = merged[feature_cols].fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Isolation Forest ---
# contamination = expected proportion of anomalies
# Using 10% since we expect ~1 in 10 trading days to have manipulation signals
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.10,
    random_state=42
)
iso_forest.fit(X_scaled)

# Predict: -1 = anomaly, 1 = normal
predictions = iso_forest.predict(X_scaled)
anomaly_scores = iso_forest.decision_function(X_scaled)

merged["iso_pred"] = predictions
merged["iso_score"] = anomaly_scores
merged["is_cross_modal_anomaly"] = (predictions == -1)

print(f"\nTotal days: {len(merged)}")
print(f"Anomaly days flagged: {(predictions == -1).sum()} ({(predictions == -1).mean()*100:.1f}%)")

# Show top anomaly days
print("\nTop 10 most anomalous days:")
top_anomalies = merged.sort_values("iso_score").head(10)
print(top_anomalies[["date", "suspicious_ratio", "avg_urgency",
                       "avg_volume_zscore", "anomaly_ratio",
                       "iso_score"]].to_string(index=False))

# Feature importance proxy — correlation with anomaly score
print("\nFeature correlations with anomaly score (higher = more important):")
for col in feature_cols:
    corr = merged[col].corr(merged["iso_score"])
    print(f"  {col}: {corr:.3f}")

# Save results
merged.to_csv("data/cross_modal_anomalies.csv", index=False)
print("\nSaved: data/cross_modal_anomalies.csv")