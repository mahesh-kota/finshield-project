from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(
    title="FinShield API",
    description="Financial Misinformation Detection System for Indian Equity Markets",
    version="1.0.0"
)

# Load data at startup
print("Loading FinShield data...")
features_df = pd.read_csv("data/features.csv")
market_df = pd.read_csv("data/market_features.csv")
anomalies_df = pd.read_csv("data/cross_modal_anomalies.csv")
labelled_df = pd.read_csv("data/telegram_posts_labelled.csv")

market_df["Date"] = pd.to_datetime(market_df["Date"])
anomalies_df["date"] = pd.to_datetime(anomalies_df["date"])
features_df["date"] = pd.to_datetime(features_df["date"], utc=True).dt.tz_localize(None)

print("Data loaded successfully.")

# Response models
class RiskResponse(BaseModel):
    ticker: str
    risk_score: float
    risk_level: str
    anomaly_days_last_90: int
    latest_volume_zscore: float
    latest_abnormal_return: float
    social_suspicious_ratio: float
    alert: bool
    message: str

class SystemStats(BaseModel):
    total_posts_analysed: int
    suspicious_posts: int
    suspicious_pct: float
    anomaly_days_flagged: int
    stocks_monitored: int
    muril_f1: float
    granger_p_value: float

class PostClassification(BaseModel):
    text: str
    predicted_class: int
    predicted_label: str
    urgency_score: float
    sentiment: float

# Routes
@app.get("/", summary="Health check")
def root():
    return {
        "status": "FinShield API is running",
        "version": "1.0.0",
        "endpoints": ["/risk/{ticker}", "/stats", "/recent-alerts", "/docs"]
    }

@app.get("/risk/{ticker}", response_model=RiskResponse, summary="Get risk score for a stock")
def get_risk(ticker: str):
    ticker_ns = ticker.upper() + ".NS" if not ticker.endswith(".NS") else ticker.upper()

    stock_data = market_df[market_df["ticker"] == ticker_ns]
    if stock_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker {ticker_ns} not found in universe. Available: {sorted(market_df['ticker'].unique().tolist())}"
        )

    # Last 90 days
    recent = stock_data.tail(90)
    anomaly_days = int(recent["is_anomaly"].sum())
    latest = recent.iloc[-1]
    vol_zscore = float(latest["volume_zscore"]) if not pd.isna(latest["volume_zscore"]) else 0.0
    abn_return = float(latest["abnormal_return"]) if not pd.isna(latest["abnormal_return"]) else 0.0

    # Social suspicious ratio (last available)
    if not anomalies_df.empty:
        latest_social = anomalies_df.sort_values("date").iloc[-1]
        susp_ratio = float(latest_social["suspicious_ratio"])
    else:
        susp_ratio = 0.0

    # Compute composite risk score (0-100)
    # Weighted combination of market and social signals
    market_score = min(anomaly_days / 10 * 40, 40)  # max 40 points from market
    vol_score = min(max(vol_zscore, 0) / 3 * 30, 30)  # max 30 points from volume
    social_score = susp_ratio * 30  # max 30 points from social

    risk_score = round(market_score + vol_score + social_score, 1)

    if risk_score >= 60:
        risk_level = "HIGH"
        alert = True
        message = f"HIGH RISK: {ticker_ns} shows elevated market anomalies ({anomaly_days} days) and high suspicious social activity ({susp_ratio*100:.0f}%). Recommend immediate review."
    elif risk_score >= 35:
        risk_level = "MEDIUM"
        alert = False
        message = f"MEDIUM RISK: {ticker_ns} shows moderate signals. Monitor closely."
    else:
        risk_level = "LOW"
        alert = False
        message = f"LOW RISK: {ticker_ns} shows normal market behaviour."

    return RiskResponse(
        ticker=ticker_ns,
        risk_score=risk_score,
        risk_level=risk_level,
        anomaly_days_last_90=anomaly_days,
        latest_volume_zscore=round(vol_zscore, 3),
        latest_abnormal_return=round(abn_return, 4),
        social_suspicious_ratio=round(susp_ratio, 3),
        alert=alert,
        message=message
    )

@app.get("/stats", response_model=SystemStats, summary="Get system-wide statistics")
def get_stats():
    suspicious = int((labelled_df["label"] == 2).sum())
    total = len(labelled_df)
    return SystemStats(
        total_posts_analysed=total,
        suspicious_posts=suspicious,
        suspicious_pct=round(suspicious / total * 100, 1),
        anomaly_days_flagged=int(anomalies_df["is_cross_modal_anomaly"].sum()),
        stocks_monitored=int(market_df["ticker"].nunique()),
        muril_f1=0.92,
        granger_p_value=0.0036
    )

@app.get("/recent-alerts", summary="Get most recent cross-modal anomaly days")
def get_recent_alerts():
    alerts = anomalies_df[anomalies_df["is_cross_modal_anomaly"] == True].copy()
    alerts["date"] = alerts["date"].dt.strftime("%Y-%m-%d")
    alerts = alerts.sort_values("date", ascending=False).head(5)
    return {
        "total_alerts": len(anomalies_df[anomalies_df["is_cross_modal_anomaly"] == True]),
        "recent_alerts": alerts[[
            "date", "suspicious_ratio", "avg_urgency",
            "avg_volume_zscore", "anomaly_ratio", "iso_score"
        ]].to_dict(orient="records")
    }

@app.get("/stocks", summary="List all monitored stocks")
def get_stocks():
    stock_summary = market_df.groupby("ticker").agg(
        anomaly_days=("is_anomaly", "sum"),
        avg_volume_zscore=("volume_zscore", "mean"),
        avg_abnormal_return=("abnormal_return", "mean")
    ).reset_index()
    stock_summary = stock_summary.sort_values("anomaly_days", ascending=False)
    return {
        "total_stocks": len(stock_summary),
        "stocks": stock_summary.round(3).to_dict(orient="records")
    }