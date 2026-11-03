> ⚠️ **CORRECTION (Nov 3, 2026):** Metrics in this document were found to be incorrect on re-measurement. MuRIL v3 actual result: macro-F1 0.570, neutral F1 0.000. SEBI backtest superseded. Granger causality corrected to null result. See FAILURES.md for details.

**# FinShield — Performance Summary**

**\*\*Data Science Semester Project | Team 4 | Amrita Sai Institute\*\***

**\*Generated: July 2026 | GitHub: github.com/mahesh-kota/finshield-project\***



**---**



**## System Overview**



**FinShield is a cross-modal financial misinformation detection system for Indian equity markets. It fuses Telegram social media NLP signals with NSE/BSE market microstructure anomalies to detect pump-and-dump manipulation before regulatory action.**



**---**



**## Stage 2: NLP Classification (MuRIL v3)**



**| Metric | MuRIL v1 | MuRIL v2 | MuRIL v3 (Final) |**

**|---|---|---|---|**

**| Training posts | 100 | 139 | 179 |**

**| Validation posts | 25 | 35 | 45 |**

**| Epochs | 5 | 10 | 10 |**

**| Class weighting | No | Yes | Yes |**

| Macro-F1 | — | — | **0.570** |

| Neutral F1 | — | — | **0.000** |

**| Neutral F1 | 0.00 | 0.73 | \*\*0.73\*\* |**

**| OpenTip F1 | 0.96 | 1.00 | \*\*1.00\*\* |**

| Suspicious F1 | — | — | **0.897** |



**\*\*Key improvement:\*\* Balanced class weights \[Neutral: 3.14, OpenTip: 1.30, Suspicious: 0.52] resolved minority class (Neutral) F1 from 0.00 to 0.73.**



**---**



**## Stage 3: Cross-Modal Anomaly Detection**



**### Granger Causality Results**



**| Dataset size | Lag 1 p-value | Lag 2 p-value | Conclusion |**

**|---|---|---|---|**

**| 174 posts (73 days) | \*\*0.0036 \*\*\* | \*\*0.0348 \*\*\* | Social leads market 1-2 days |**

**| 224 posts (75 days) | 0.0654 | 0.2108 | Marginally significant |**



**Note:** Granger result was corrected Nov 2, 2026. Re-analysis with proper stationarity checks showed no robust evidence that social signals lead market anomalies. Reported as a null result.



**### Isolation Forest Results**



**| Metric | Value |**

**|---|---|**

**| Overlapping social+market days | 75 |**

**| Anomaly days flagged | 8 (10.7%) |**

**| Top feature (SHAP) | avg\_urgency (0.457) |**

**| Second feature (SHAP) | anomaly\_ratio (0.321) |**

**| Social vs market signal balance | 4 social features, 4 market features |**



**### SHAP Feature Importance**



**| Rank | Feature | SHAP Value | Signal Type |**

**|---|---|---|---|**

**| 1 | avg\_urgency | 0.457 | Social |**

**| 2 | anomaly\_ratio | 0.321 | Market |**

**| 3 | avg\_range\_zscore | 0.293 | Market |**

**| 4 | avg\_abnormal\_return | 0.277 | Market |**

**| 5 | muril\_confidence | 0.251 | Social |**

**| 6 | avg\_volume\_zscore | 0.240 | Market |**

**| 7 | avg\_sentiment | 0.196 | Social |**

**| 8 | suspicious\_ratio | 0.132 | Social |**



**---**



**## Stage 4: SEBI Backtest Evaluation**



**| Case | Penalty | Detected | Early Detection |**

**|---|---|---|---|**

**| Sadhana-Sharpline (YouTube P\&D) | ₹41.85 Cr | ❌ MISSED | N/A (stock not in universe) |**

**| Retro Green Revolution (Telegram) | ₹2.8 Cr | ✅ DETECTED | 954 days before SEBI |**

**| Mauria Udyog (Mass SMS) | ₹47.7 Cr | ✅ DETECTED | 1075 days before SEBI |**

**| Sharpline Broadcast (Social Media) | ₹5.2 Cr | ✅ DETECTED | \*\*11 days before SEBI\*\* |**



**| Evaluation Metric | Value |**

**|---|---|**

**| Recall | Superseded — see Sadhna case limitation (Nov 3) |**

**| Precision@K (K=8) | 0.38 |**

**| Best early detection | 11 days (Sharpline case) |**



**---**



**## Dataset Summary**



**| Component | Count |**

**|---|---|**

**| Telegram channels scraped | 9 public channels |**

**| Raw messages collected | 274 |**

**| Clean messages (after filtering) | 274 |**

**| Manually annotated posts | 224 |**

**| Neutral (class 0) | 24 (10.7%) |**

**| Open Trading Tip (class 1) | 57 (25.4%) |**

**| Suspicious/P\&D (class 2) | 143 (63.8%) |**

**| NSE stocks monitored | 47 mid/small-cap |**

**| Market data rows | 50,571 |**

**| Date range | Jan 2022 – Jul 2026 |**

**| Cross-modal overlap days | 75 |**



**---**



**## Novel Research Contributions**



**1. \*\*First cross-modal social-market anomaly detection pipeline for Indian NSE/BSE equities\*\* — all prior work focuses on cryptocurrency markets**

**2. \*\*Empirical linguistic finding\*\* — Indian financial Telegram broadcast channels are English-dominant (<1% Hinglish), contrary to prior assumptions**

**3. \*\*Reproducible SEBI-backtest methodology\*\* — using real enforcement orders as ground truth, usable as benchmark for future work**

**4. \*\*Coordinated inauthentic behaviour detection\*\* — NetworkX graph identified 2-channel coordinated community sharing 9 near-duplicate messages**



**---**



**## Live System URLs**



**| Component | URL |**

**|---|---|**

**| Streamlit Dashboard | https://finshield-project-y2shew3jwg2qcjf5yizmx7.streamlit.app |**

**| GitHub Repository | https://github.com/mahesh-kota/finshield-project |**

**| FastAPI Docs | http://localhost:8502/docs (run locally) |**

