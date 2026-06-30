# FinShield

**A Cross-Modal Financial Misinformation Detection System for Indian Equity Markets Using Hinglish NLP and Temporal Anomaly Fusion**

A semester project by a 4-member Data Science team detecting coordinated pump-and-dump misinformation in Indian retail investor communities (Telegram, Reddit) and correlating it with abnormal NSE/BSE market activity.

---

## Problem Statement

Indian retail investors are increasingly targeted by coordinated pump-and-dump schemes spread through Telegram and WhatsApp — largely in **Hinglish** (code-switched Hindi-English), a language style most existing NLP models fail to handle. No current system jointly correlates this social signal with real market anomalies for Indian equities. FinShield aims to close that gap with a system that can flag suspicious activity *before or during* a manipulation event, validated against real SEBI enforcement orders.

## Project Status

In progress — Day 12 of a 7-month (210 day) build. See `docs/work_log.xlsx` for the daily task tracker and `docs/weekly_notes.md` for weekly summaries.

## Team

| Role | Member | Responsibility |
|---|---|---|
| Team Lead | Mahesh | Coordination, all 4 roles below |
| M1 - Data Engineer | - | Scraping, data pipelines, storage |
| M2 - NLP Engineer | - | Hinglish preprocessing, MuRIL fine-tuning |
| M3 - Market Modeller | - | Anomaly detection, TFT fusion model, Granger causality |
| M4 - Graph & Dashboard | - | Network analysis, Streamlit dashboard, evaluation |

## System Pipeline
## Tech Stack

- **Data collection**: Telethon (Telegram), PRAW (Reddit), yfinance (NSE/BSE)
- **Data storage**: CSV / Parquet files (local) — chosen over MongoDB after evaluating both; project's data scale doesn't require a database, and local storage is more reliable for demo purposes
- **NLP**: HuggingFace Transformers, MuRIL, IndicNLP, spaCy
- **ML**: PyTorch, scikit-learn, PyTorch Forecasting (TFT), statsmodels, NetworkX
- **Dashboard**: Streamlit, Plotly, FastAPI, SHAP

## Folder Structure
## Setup Instructions

```bash
git clone https://github.com/mahesh-kota/finshield-project.git
cd finshield-project
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

> GPU-only step: MuRIL fine-tuning requires a GPU. Use Google Colab (free T4 GPU) for that stage; everything else runs locally on a standard laptop.

## Data Sources

- **Telegram**: Public finance/trading-tip channels, scraped via Telethon (active, working)
- **Reddit**: r/IndiaInvestments and similar, via PRAW (pending academic API approval)
- **Twitter/X**: Evaluated and excluded due to restrictive free-tier API access for academic developer accounts as of 2026; noted as future work
- **Market data**: NSE/BSE OHLCV data via yfinance, covering 50 mid/small-cap stocks (see `data/stock_universe.txt`)

## Ground Truth / Evaluation

Model alerts are backtested against publicly available SEBI enforcement orders (2019-2024) involving market manipulation, used as labelled ground truth for evaluating early detection performance.

## License

Academic semester project - for educational use only. Not financial advice, not for production trading use.
