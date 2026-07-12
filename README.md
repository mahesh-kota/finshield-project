# FinShield

**A Cross-Modal Financial Misinformation Detection System for Indian Equity Markets Using Hinglish NLP and Temporal Anomaly Fusion**

A semester project by a 4-member Data Science team detecting coordinated pump-and-dump misinformation in Indian retail investor communities (Telegram) and correlating it with abnormal NSE/BSE market activity.

---

## Results Summary

| Metric | Value |
|---|---|
| MuRIL Classifier Accuracy | 91.4% |
| Weighted F1 Score | 0.92 |
| Neutral Class F1 (after class weighting) | 0.73 |
| Granger Causality p-value (lag 1) | 0.004 *** |
| Social signal lead time | 1–2 trading days |
| SEBI backtest recall | 75% (3/4 cases) |
| Precision@8 | 0.38 |
| Earliest detection before SEBI | 11 days (Sharpline case) |
| Stocks monitored | 47 NSE mid/small-cap |
| Posts analysed | 174 (9 public Telegram channels) |
| Anomaly days flagged | 8 cross-modal anomaly days |

---

## Problem Statement

Indian retail investors are increasingly targeted by coordinated pump-and-dump schemes spread through Telegram and WhatsApp. These schemes use aggressive, hype-driven language to artificially inflate mid/small-cap stock prices, causing significant financial losses to uninformed investors. No current system jointly correlates social media manipulation signals with real NSE/BSE market anomalies in real time. FinShield addresses this gap with a system that can flag suspicious activity before or during a manipulation event, validated against real SEBI enforcement orders.

## Key Research Finding

Empirical Granger causality testing (p=0.004 at lag 1 after first-differencing) confirms that suspicious social media activity on Indian financial Telegram channels **statistically precedes market anomalies by 1–2 trading days**. This is the first formal quantitative demonstration of this causal relationship for Indian NSE/BSE equities.

A secondary empirical finding: Indian financial tip channels on Telegram are English-dominant (<1% genuine Hindi/Hinglish code-switching), challenging the common assumption that Indian financial social media content is primarily Hinglish.

## System Pipeline

## Tech Stack

| Layer | Tools |
|---|---|
| Data collection | Telethon, yfinance |
| Data storage | CSV / Pandas (local) |
| NLP | MuRIL (google/muril-base-cased), VADER, spaCy |
| ML | PyTorch, scikit-learn (Isolation Forest), statsmodels (Granger) |
| Dashboard | Streamlit, Plotly |
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
            streamlit plotly statsmodels networkx fastapi uvicorn
python -m spacy download en_core_web_sm
```

Create a `.env` file:

Run the dashboard:
```bash
streamlit run dashboard/app.py
```

> GPU required only for MuRIL fine-tuning — use Google Colab free T4. All other steps run locally on a standard laptop (tested on Dell Vostro 14, i5-1334U, 8GB RAM, Windows 11).

## Limitations & Future Work

- Dataset size: 174 labelled posts — expand to 500+ for more robust training
- Stock universe: 47 stocks — expand to broader NSE500 for wider coverage
- Hinglish content: <1% in broadcast channels — future work to include discussion groups
- Twitter/X excluded due to API access restrictions — future data source
- SEBI backtest limited to 4 cases — expand ground truth as more orders are published

## License

Academic semester project — for educational use only. Not financial advice. Not for production or commercial deployment.

