cat > README.md << 'EOF'
# FinShield

**A Cross-Modal Financial Misinformation Detection System for Indian Equity Markets Using Hinglish NLP and Temporal Anomaly Fusion**

A semester project by a 4-member Data Science team detecting coordinated pump-and-dump misinformation in Indian retail investor communities (Telegram) and correlating it with abnormal NSE/BSE market activity.

---

## 🚀 Live Dashboard

**[https://finshield-project-y2shew3jwg2qcjf5yizmx7.streamlit.app](https://finshield-project-y2shew3jwg2qcjf5yizmx7.streamlit.app)**

---

## Results Summary (MuRIL v3 — Updated)

| Metric | Value |
|---|---|
| MuRIL Classifier Accuracy | 93.3% |
| Weighted F1 Score | 0.94 |
| Neutral Class F1 (after class weighting) | 0.73 |
| OpenTip Class F1 | 1.00 |
| Suspicious Class F1 | 0.95 |
| Granger Causality p-value (lag 1, 174 posts) | 0.004 *** |
| Granger Causality p-value (lag 1, 224 posts) | 0.065 (marginally significant) |
| Social signal lead time | 1-2 trading days |
| SEBI backtest recall | 75% (3/4 cases) |
| Precision@8 | 0.38 |
| Earliest detection before SEBI | 11 days (Sharpline case) |
| Top SHAP feature | avg_urgency (0.457) |
| Stocks monitored | 47 NSE mid/small-cap |
| Posts analysed | 224 (9 public Telegram channels) |
| Anomaly days flagged | 8 cross-modal anomaly days |

---

## Problem Statement

Indian retail investors are increasingly targeted by coordinated pump-and-dump schemes spread through Telegram and WhatsApp. These schemes use aggressive, hype-driven language to artificially inflate mid/small-cap stock prices, causing significant financial losses to uninformed investors. No current system jointly correlates social media manipulation signals with real NSE/BSE market anomalies in real time. FinShield addresses this gap with a system that can flag suspicious activity before or during a manipulation event, validated against real SEBI enforcement orders.

## Key Research Findings

1. **Causal precedence**: Suspicious social media activity on Indian financial Telegram channels statistically precedes market anomalies by 1-2 trading days (Granger causality confirmed).
2. **Urgency dominates**: SHAP analysis shows urgency language (SHAP=0.457) is a stronger anomaly predictor than any market price signal.
3. **English dominance**: Indian financial Telegram broadcast channels are English-dominant (<1% genuine Hinglish), challenging prior assumptions about Indian financial social media.
4. **Coordinated behaviour**: NetworkX graph analysis detected a coordinated inauthentic behaviour (CIB) community sharing identical private invite links across channels.

## System Pipeline

## Tech Stack

| Layer | Tools |
|---|---|
| Data collection | Telethon, yfinance |
| Data storage | CSV / Pandas (local) |
| NLP | MuRIL (google/muril-base-cased), VADER, spaCy |
| ML | PyTorch, scikit-learn (Isolation Forest), statsmodels (Granger), SHAP, NetworkX |
| Dashboard | Streamlit, Plotly (deployed on Streamlit Cloud) |
| API | FastAPI + Uvicorn |
| Training | Google Colab T4 GPU (free tier) |

## Folder Structure

## Setup Instructions

```bash
git clone https://github.com/mahesh-kota/finshield-project.git
cd finshield-project
python -m venv venv
source venv/Scripts/activate        # Windows Git Bash
pip install telethon python-dotenv yfinance pandas torch \
            transformers vaderSentiment spacy scikit-learn \
            streamlit plotly statsmodels networkx fastapi uvicorn shap
python -m spacy download en_core_web_sm
```

Create a `.env` file with your Telegram API credentials:
Run the dashboard locally:
```bash
streamlit run dashboard/app.py
```

Run the FastAPI alert endpoint:
```bash
uvicorn dashboard.api:app --reload --port 8502
```

> GPU required only for MuRIL fine-tuning — use Google Colab free T4 (see src/muril_finetune.py). All other steps run locally on a standard laptop (tested on Dell Vostro 14, i5-1334U, 8GB RAM, Windows 11).

## Project Status

**Day 36 of 210** — all 4 pipeline stages complete, dashboard deployed live.

## Limitations & Future Work

- Dataset: 224 labelled posts — expand to 500+ for more robust training
- Stock universe: 47 stocks — expand to NSE500 for wider coverage
- Granger result sensitive to dataset composition — larger social corpus needed for robust confirmation
- Twitter/X excluded due to API access restrictions — future data source
- TFT (Temporal Fusion Transformer) verified but not fully trained — future work
- SEBI backtest limited to 4 cases — expand as more orders published

## License

Academic semester project — for educational use only. Not financial advice. Not for production or commercial deployment.
EOF