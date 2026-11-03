# Contributors

## Team 4 — FinShield: Financial Misinformation Detection System

| Member | Roll Number | Module | Contribution |
|---|---|---|---|
| Mahesh Rao Kota | 23AJ1A4442 | M1 — Data Eng (primary author) | Telegram scraper, channel selection, 224-post manual labeling, data pipeline, MuRIL fine-tuning, TF-IDF baseline, market anomaly detection, SEBI case validation, Streamlit dashboard, FastAPI endpoints, documentation, corrections |
| Anusha Guntaka | 23AJ1A4429 | M2 — NLP Eng | [fill in: what did she actually do?] |
| Saketh Gunda | 23AJ1A4427 | M3 — Market Model | [fill in: what did he actually do?] |
| Jeevan Kesava Reddy Avula | 23AJ1A4406 | M4 — Graph/Dash | [fill in: what did he actually do?] |

## Module breakdown

- **M1 — Data Eng:** Collection and preparation. Telegram scraper (5 public channels), deduplication, min-length filtering, manual 3-class annotation (Neutral / OpenTip / Suspicious), CSV/pandas storage.
- **M2 — NLP Eng:** Text classification and feature extraction. MuRIL-base fine-tuning with class-weighted loss, TF-IDF + logistic regression baseline, VADER sentiment, urgency scoring, ticker NER, price target regex.
- **M3 — Market Model:** Market-side anomaly detection. yfinance OHLCV pipeline for 50 NSE mid/small-cap stocks, abnormal return (CAPM), volume z-score, Isolation Forest cross-modal fusion, SEBI case backtest.
- **M4 — Graph/Dash:** Visualization and delivery. NetworkX channel-coordination graph, near-duplicate detection, Streamlit dashboard (price charts, anomaly heatmap, flagged posts), FastAPI risk-scoring endpoints, SHAP feature importance plots.

## Notes on contribution

This project was primarily authored by Mahesh Rao Kota. The commit history reflects this — all commits are from Mahesh. Teammates contributed to [fill in: e.g., "labeling sessions on Day 15 and Day 31", "paper review", "dashboard testing", "literature review on Days 2-3"]. The module breakdown above reflects the division of work agreed at project start; the actual implementation was concentrated in one author.

## Corrections and revisions

On Nov 2–3, 2026, the team re-ran all evaluation scripts and found that several previously reported metrics were incorrect (MuRIL v3 F1, SEBI backtest recall, Granger causality p-value). All corrections are documented in `docs/FAILURES.md` and marked inline in `docs/performance_summary.md` and `docs/weekly_notes.md`. The underlying work — data collection, labeling, feature engineering, dashboard, API — is unchanged and verified.
