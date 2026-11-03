# What Didn't Work

Honest log of approaches that failed, why, and what we learned.

## MuRIL v1 (Jul 8, 2026)
- **What we tried:** Fine-tune MuRIL-base on 100 labeled posts, 5 epochs.
- **What happened:** Neutral class F1 = 0.00. Model never predicted neutral.
- **Why:** 13 neutral examples in training set. Not enough to learn from.
- **What we learned:** Class imbalance needs explicit handling, not just more epochs.

## MuRIL v2 (Jul 10, 2026)
- **What we tried:** Class-weighted loss, 10 epochs, 174 labels.
- **What we reported at the time:** Weighted F1 0.92, neutral F1 improved to 0.73.
- **What actually happened (re-measured Nov 3):** Neutral F1 = 0.000. The 0.73 was from a buggy evaluation script that computed F1 on the training set, not validation.
- **Why:** Evaluation bug. We didn't catch it until we re-ran with a clean script.
- **What we learned:** Always verify metrics against a fresh run before reporting.

## MuRIL v3 (Nov 3, 2026)
- **What we tried:** Class-weighted loss, 224 labels, 5 epochs, corrected evaluation.
- **What happened:** Macro-F1 0.570. Neutral F1 = 0.000. Open_tip 0.815, suspicious 0.897.
- **Why:** 60 gradient steps total is far below what a transformer needs. Also, 4 of 5 misclassified neutrals were borderline cases containing advice or promo language — the label definition was too permissive.
- **What we learned:** Label quality, not model capacity, is the bottleneck.

## TF-IDF + LR baseline (Nov 3, 2026)
- **What we tried:** TF-IDF (1-2 grams, 5000 features) + logistic regression with balanced class weights.
- **What happened:** Macro-F1 0.844. Beat MuRIL by 0.27.
- **Why:** Hinglish financial text has strong lexical signals (buy, target, SL, join, DM). Below ~1,000 examples, lexical models outperform transformers.
- **What we learned:** Always run a simple baseline before investing in a transformer.

## SEBI Sadhna case validation (Nov 3, 2026)
- **What we tried:** Test the market anomaly detector against SEBI's examination period (27 Apr - 30 Sep 2022).
- **What happened:** Detector showed no lift. Anomaly share was 5.3% pre-period, 7.1% patch 1, 7.5% patch 2 — noise, not signal.
- **Why:** Three detectors failed. Rolling z-score adapts to the ramp. Trailing median lags but catches up. Fixed 2021 baseline is unavailable (zero volume in 2021).
- **What we learned:** Unsupervised threshold detection is poorly suited to gradual-onset manipulation. Change-point detection (CUSUM, PELT) is the right tool.

## Granger causality (Nov 2, 2026)
- **What we tried:** Test whether social signals lead market anomalies.
- **What we reported at the time:** p=0.004 at lag 1, social leads market by 1-2 days.
- **What actually happened (re-analysis):** No robust evidence after proper stationarity checks.
- **Why:** First-differencing removed the signal. Original test was on non-stationary series.
- **What we learned:** Stationarity checks are not optional.

## What we'd do differently
- Tighten the neutral label definition before collecting more data
- Validate on a second SEBI case before claiming recall
- Run TF-IDF baseline before fine-tuning MuRIL
- Write evaluation scripts that load from disk, not from memory
