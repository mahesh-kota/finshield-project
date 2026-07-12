import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(
    page_title="FinShield",
    page_icon="🛡️",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    features = pd.read_csv("data/features.csv")
    market = pd.read_csv("data/market_features.csv")
    anomalies = pd.read_csv("data/cross_modal_anomalies.csv")
    labelled = pd.read_csv("data/telegram_posts_labelled.csv")
    return features, market, anomalies, labelled

features_df, market_df, anomalies_df, labelled_df = load_data()

# Convert dates
market_df["Date"] = pd.to_datetime(market_df["Date"])
anomalies_df["date"] = pd.to_datetime(anomalies_df["date"])
features_df["date"] = pd.to_datetime(features_df["date"], utc=True).dt.tz_localize(None)

# Header
st.title("🛡️ FinShield")
st.markdown("**Financial Misinformation Detection System for Indian Equity Markets**")
st.divider()

# Sidebar
st.sidebar.title("Controls")
selected_stock = st.sidebar.selectbox(
    "Select Stock",
    sorted(market_df["ticker"].unique())
)
date_range = st.sidebar.slider(
    "Market data days",
    min_value=30,
    max_value=365,
    value=90
)

# Top metrics row
col1, col2, col3, col4 = st.columns(4)
total_posts = len(labelled_df)
suspicious_posts = (labelled_df["label"] == 2).sum()
anomaly_days = anomalies_df["is_cross_modal_anomaly"].sum()
stocks_monitored = market_df["ticker"].nunique()

col1.metric("Posts Analysed", total_posts)
col2.metric("Suspicious Posts", f"{suspicious_posts} ({suspicious_posts/total_posts*100:.0f}%)")
col3.metric("Anomaly Days Flagged", int(anomaly_days))
col4.metric("Stocks Monitored", stocks_monitored)

st.divider()

# Row 1: Stock price chart + anomaly overlay
st.subheader(f"📈 Price & Volume Anomalies — {selected_stock}")

stock_data = market_df[market_df["ticker"] == selected_stock].tail(date_range)
anomaly_days_stock = stock_data[stock_data["is_anomaly"] == True]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=stock_data["Date"],
    y=stock_data["Close"],
    mode="lines",
    name="Close Price",
    line=dict(color="#534AB7", width=2)
))

fig.add_trace(go.Scatter(
    x=anomaly_days_stock["Date"],
    y=anomaly_days_stock["Close"],
    mode="markers",
    name="Anomaly Day",
    marker=dict(color="red", size=10, symbol="x")
))

fig.update_layout(
    height=350,
    showlegend=True,
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig, use_container_width=True)

# Row 2: Social signals + cross-modal anomalies
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Social Signal Timeline")
    fig2 = px.bar(
        anomalies_df.sort_values("date"),
        x="date",
        y="suspicious_ratio",
        color="is_cross_modal_anomaly",
        color_discrete_map={True: "#FF4B4B", False: "#534AB7"},
        labels={"suspicious_ratio": "Suspicious Post Ratio", "date": "Date"},
        title="Daily Suspicious Post Ratio (Red = Cross-Modal Anomaly)"
    )
    fig2.update_layout(
        height=300,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    st.subheader("🔥 Market Anomaly Heatmap")
    top_stocks = market_df.groupby("ticker")["is_anomaly"].sum().sort_values(
        ascending=False).head(10).reset_index()
    top_stocks.columns = ["Stock", "Anomaly Days"]

    fig3 = px.bar(
        top_stocks,
        x="Anomaly Days",
        y="Stock",
        orientation="h",
        color="Anomaly Days",
        color_continuous_scale="Reds",
        title="Top 10 Most Anomalous Stocks"
    )
    fig3.update_layout(
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# Row 3: MuRIL classification breakdown + recent flagged posts
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🤖 MuRIL Classification Results")
    label_counts = labelled_df["label"].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]
    label_counts["Label"] = label_counts["Label"].map(
        {0: "Neutral", 1: "Open Tip", 2: "Suspicious"})

    fig4 = px.pie(
        label_counts,
        values="Count",
        names="Label",
        color="Label",
        color_discrete_map={
            "Neutral": "#1D9E75",
            "Open Tip": "#534AB7",
            "Suspicious": "#FF4B4B"
        },
        title=f"174 Posts | Weighted F1 = 0.92 | Accuracy = 91.4%"
    )
    fig4.update_layout(height=300)
    st.plotly_chart(fig4, use_container_width=True)

with col_b:
    st.subheader("🚨 Recent Flagged Posts")
    suspicious = labelled_df[labelled_df["label"] == 2][
        ["channel", "date", "text"]].tail(8)
    suspicious["text"] = suspicious["text"].str[:80] + "..."
    suspicious["date"] = pd.to_datetime(
        suspicious["date"], utc=True).dt.strftime("%Y-%m-%d")
    st.dataframe(
        suspicious,
        use_container_width=True,
        height=280,
        hide_index=True
    )

st.divider()

# Row 4: Cross-modal anomaly table
st.subheader("⚠️ Cross-Modal Anomaly Days (Social + Market signals combined)")
anomaly_display = anomalies_df[anomalies_df["is_cross_modal_anomaly"] == True][[
    "date", "suspicious_ratio", "avg_urgency",
    "avg_volume_zscore", "anomaly_ratio", "iso_score"
]].copy()
anomaly_display.columns = [
    "Date", "Suspicious Ratio", "Avg Urgency",
    "Volume Z-Score", "Market Anomaly Ratio", "Isolation Score"
]
anomaly_display["Date"] = pd.to_datetime(
    anomaly_display["Date"]).dt.strftime("%Y-%m-%d")
for col in ["Suspicious Ratio", "Avg Urgency", "Volume Z-Score",
            "Market Anomaly Ratio", "Isolation Score"]:
    anomaly_display[col] = anomaly_display[col].round(3)

st.dataframe(anomaly_display, use_container_width=True, hide_index=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align:center;color:gray;font-size:12px'>
FinShield | Data Science Semester Project | MuRIL F1=0.92 |
Granger Causality: p=0.004 (lag 1) | 47 NSE Stocks Monitored
</div>
""", unsafe_allow_html=True)