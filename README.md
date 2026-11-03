# FinShield

**A Cross-Modal Financial Misinformation Detection System for Indian Equity Markets Using Hinglish NLP and Temporal Anomaly Fusion**

A semester project by a 4-member Data Science team studying coordinated pump-and-dump misinformation in Indian retail investor communities (Telegram) and its relationship to NSE/BSE market activity.

---

## Live Dashboard

**https://finshield-project-y2shew3jwg2qcjf5yizmx7.streamlit.app**

---

## Results Summary

| Metric | Value |
|---|---|
| MuRIL v3 macro-F1 (45-post validation set) | 0.570 |
| MuRIL v3 neutral F1 | 0.000 |
| TF-IDF+LR baseline macro-F1 | 0.844 |
| Neutral / OpenTip / Suspicious F1 | 0.73 / 1.00 / 0.95 |
| Granger causality, lag 1 (174 posts) | p = 0.004 (exploratory, not robust) |
| Granger causality, lag 1 (224 posts) | p = 0.065 (not significant) |
| Event study, stocks (53 event days) | No significant effect |
| Event study, NIFTY/BANKNIFTY (up to 169 event days) | No significant lead after multiple-testing correction |
| Market-only TFT (next-day abnormal return), test MAE | 0.0104 vs 0.0116 (5-day rolling mean) vs 0.0141 (naive) |
| Top SHAP feature in Isolation Forest | avg_urgency (mean abs SHAP ~0.47) |
| SEBI backtest (preliminary, window-based) | 3 of 4 cases had a flag inside the investigation window |
| Stocks monitored | 47 NSE mid/small-cap |
| Labelled posts | 224 (8 public Telegram channels) |
| Additional scraped, unlabelled messages | 3,410 (461 distinct post days) |

---

## Problem Statement

Indian retail investors are increasingly targeted by coordinated pump-and-dump schemes spread through Telegram and WhatsApp. These schemes use aggressive, hype-driven language to push up mid/small-cap stock prices, causing financial losses to uninformed investors. FinShield classifies tip-channel posts with a fine-tuned MuRIL model and examines whether that social signal is related to anomalies in NSE/BSE market data.

## Key Findings

1. **NLP classification is partially working**: MuRIL v3 separates Open Tip (F1 0.815) and Suspicious (F1 0.897) well, but never predicts Neutral (F1 0.000). A TF-IDF + logistic regression baseline achieves macro-F1 0.844, outperforming MuRIL's 0.570. Diagnosis: label quality, not model capacity, is the bottleneck — 4 of 5 misclassified neutrals were borderline cases containing advice or promotional language.
2. **Social-to-market link is not confirmed (exploratory)**: An initial Granger test on 174 posts suggested suspicious-post activity preceded market anomalies (p = 0.004, lag 1). The result was not robust: at 224 posts it weakened to p = 0.065, and event studies on stocks (53 event days) and on NIFTY/BANKNIFTY (up to 169 event days, 3,410 scraped messages) found no significant lead after multiple-testing correction. A pre-specified out-of-sample test is left for future work.
3. **Urgency language drives the anomaly flags**: Within the Isolation Forest, the urgency score contributed most to flagging the 8 anomalous days. This explains the model, and is not evidence of causation.
4. **English dominance**: In the posts sampled, genuine Hindi/Hinglish content was under 1%; trading jargon (CE/PE/SL/Target) dominates (small sample).
5. **Coordinated behaviour**: A NetworkX graph found two channels sharing near-identical posts with the same private invite links.

## System Pipeline

```
Stage 1: Data ingestion
  - Telegram public channels via Telethon
  - NSE/BSE OHLCV for 47 stocks and NIFTY/BANKNIFTY/SENSEX via yfinance

Stage 2: NLP (MuRIL v3)
  - 224 manually labelled posts, 3 classes (Neutral / OpenTip / Suspicious)
  - Fine-tuned on Google Colab T4 with balanced class weights
  - Per-post features: VADER sentiment, urgency score, ticker NER

Stage 3: Analysis
  - Market features: abnormal return, volume z-score, range z-score
  - Granger causality, event studies (stock-level and index-level)
  - Isolation Forest fusion, SHAP, NetworkX coordination graph

Stage 4: Evaluation and delivery
  - SEBI enforcement-order backtest (preliminary)
  - Streamlit dashboard (Streamlit Cloud) and FastAPI endpoints
```

## Tech Stack

| Layer | Tools |
|---|---|
| Data collection | Telethon, yfinance |
| Data storage | CSV / Pandas (local) |
| NLP | MuRIL (google/muril-base-cased), VADER, spaCy |
| ML | PyTorch, scikit-learn (Isolation Forest), statsmodels, SHAP, NetworkX, PyTorch Forecasting |
| Dashboard | Streamlit, Plotly |
| API | FastAPI + Uvicorn |
| Training | Google Colab T4 GPU (free tier) |

## Folder Structure

```
finshield-project/
├── data/                 # datasets (raw scraped files stay local)
├── src/                  # pipeline and analysis scripts
├── dashboard/            # app.py (Streamlit), api.py (FastAPI)
├── models/               # MuRIL configs and tokenizers (weights are local only)
├── docs/                 # paper draft, weekly notes, result tables, plots
├── tests/                # test and check scripts
├── annotate.py           # manual annotation tool
├── telegram_scraper.py   # Telegram scraper
└── README.md
```

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

`requirements.txt` is intentionally minimal because Streamlit Cloud only needs the dashboard packages. Use the `pip install` command above for the full pipeline.

Create a `.env` file with your Telegram API credentials:

```
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
```

Run the dashboard locally:

```bash
streamlit run dashboard/app.py
```

Run the FastAPI endpoint:

```bash
uvicorn dashboard.api:app --reload --port 8502
```

GPU is needed only for MuRIL fine-tuning; use the free Colab T4 (see `src/muril_finetune.py`). Everything else runs on a standard laptop (tested on a Dell Vostro 14, i5-1334U, 8 GB RAM, Windows 11).

## Project Status

In progress. All four pipeline stages are built and the dashboard is deployed. Current work is stress-testing the social-to-market claim and tightening the SEBI evaluation.

## Limitations and Future Work

- Social-to-market lead not confirmed: the Granger result was fragile and the event studies found no significant effect. Next step is a pre-specified test on newly collected data.
- SEBI backtest is preliminary: a case counts as detected if any flagged day falls between the investigation start and the SEBI order date, which spans years, and the SEBI-named stocks are not checked against the 47-stock universe. Only 4 cases are used.
- Dataset: 224 labelled posts from 8 channels, with few Neutral examples (24). Expand and add inter-annotator agreement.
- The 3,410 scraped messages are classified by the model, not manually labelled, so they should not be used as training data.
- Only 3 of 223 labelled posts name a stock in the universe, so stock-level tests have little data; most posts discuss NIFTY/BANKNIFTY options.
- Social data is too sparse for daily TFT covariates, so the TFT uses market features only.
- Twitter/X excluded due to API access restrictions.

## License

Academic semester project, for educational use only. Not financial advice. Not for production or commercial deployment.