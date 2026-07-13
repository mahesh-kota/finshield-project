import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Load data
social_df = pd.read_csv("data/features.csv")
market_df = pd.read_csv("data/market_features.csv")

# Aggregate social daily
social_df["date"] = pd.to_datetime(social_df["date"], utc=True).dt.date
social_daily = social_df.groupby("date").agg(
    post_count=("muril_pred", "count"),
    suspicious_ratio=("muril_pred", lambda x: (x == 2).mean()),
    avg_sentiment=("sentiment_compound", "mean"),
    avg_urgency=("urgency_score", "mean"),
    muril_confidence=("muril_confidence", "mean")
).reset_index()
social_daily["date"] = pd.to_datetime(social_daily["date"])

# Aggregate market daily
market_df["Date"] = pd.to_datetime(market_df["Date"])
market_daily = market_df.groupby("Date").agg(
    total_anomalies=("is_anomaly", "sum"),
    avg_abnormal_return=("abnormal_return", "mean"),
    avg_volume_zscore=("volume_zscore", "mean"),
    avg_range_zscore=("range_zscore", "mean"),
    anomaly_ratio=("is_anomaly", "mean")
).reset_index()

merged = pd.merge(social_daily, market_daily,
                  left_on="date", right_on="Date", how="inner")

feature_cols = [
    "suspicious_ratio", "avg_sentiment", "avg_urgency",
    "muril_confidence", "avg_abnormal_return",
    "avg_volume_zscore", "avg_range_zscore", "anomaly_ratio"
]

X = merged[feature_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

# Retrain Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.10, random_state=42)
iso_forest.fit(X_scaled)
predictions = iso_forest.predict(X_scaled)
merged["is_anomaly"] = (predictions == -1)

print(f"Total days: {len(merged)}, Anomaly days: {merged['is_anomaly'].sum()}")

# SHAP explanation
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(iso_forest)
shap_values = explainer.shap_values(X_scaled_df)

# Feature importance summary
shap_importance = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("\n=== SHAP Feature Importance ===")
for _, row in shap_importance.iterrows():
    bar = "█" * int(row["mean_abs_shap"] * 100)
    print(f"  {row['feature']:<25} {row['mean_abs_shap']:.4f}  {bar}")

# Save SHAP summary plot
print("\nSaving SHAP summary plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_scaled_df, show=False)
plt.tight_layout()
plt.savefig("docs/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: docs/shap_summary.png")

# Show SHAP for top anomaly days
anomaly_indices = merged[merged["is_anomaly"]].index.tolist()
print(f"\n=== SHAP values for {len(anomaly_indices)} anomaly days ===")
for idx in anomaly_indices[:3]:
    print(f"\nDate: {merged.loc[idx, 'date'].date()}")
    day_shap = pd.Series(shap_values[idx], index=feature_cols).sort_values(key=abs, ascending=False)
    for feat, val in day_shap.items():
        direction = "↑ pushed toward anomaly" if val < 0 else "↓ pushed toward normal"
        print(f"  {feat:<25} {val:+.4f}  {direction}")

# Save feature importance
shap_importance.to_csv("docs/shap_importance.csv", index=False)
print("\nSaved: docs/shap_importance.csv")