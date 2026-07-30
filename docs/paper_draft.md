**# FinShield: A Cross-Modal Financial Misinformation Detection System for Indian Equity Markets Using Hinglish NLP and Temporal Anomaly Fusion**



**\*\*Authors:\*\* Kota Mahesh Rao, Guntaka Anusha, Gunda Saketh, Avula Jeevan Kesava Reddy**

**\*\*Guide:\*\* E. Babitha**

**\*\*Institution:\*\* Amrita Sai Institute of Science and Technology**

**\*\*Department:\*\* Computer Science \& Engineering (Data Science)**

**\*\*Date:\*\* July 2026**



**---**



**## I. INTRODUCTION**



**India's retail equity investor base crossed 100 million SEBI-registered demat accounts by 2023, driven by increased digital access and post-COVID market participation. This expansion has been accompanied by a parallel proliferation of organised financial misinformation — coordinated campaigns to artificially inflate the price of mid-cap and small-cap stocks through mass dissemination of misleading "tip" content on Telegram, WhatsApp, and YouTube. These campaigns, commonly referred to as pump-and-dump (P\&D) schemes, exploit the information asymmetry between well-connected operators and uninformed retail investors.**



**The Securities and Exchange Board of India (SEBI) has identified an increasing number of enforcement actions related to social media-facilitated market manipulation. Notable cases include: the Sadhana-Sharpline Broadcast case (2023, ₹41.85 crore penalty), the Retro Green Revolution Telegram manipulation (2026, ₹2.8 crore), and the Mauria Udyog coordinated SMS/Telegram campaign (2026, ₹47.7 crore, 222 entities banned). In each case, SEBI's detection was retrospective — occurring months to years after the manipulation window, with lead times ranging from 258 to 2007 days between the manipulation period end and the enforcement order.**



**Existing automated detection systems face two critical limitations for the Indian context. First, a \*\*linguistic gap\*\*: standard NLP models trained on English-only or general multilingual corpora are not calibrated for Indian financial trading jargon, which uses specific shorthand (CE/PE/SL/Target) that is semantically distinct from everyday language. Second, a \*\*signal modality gap\*\*: all existing pump-and-dump detection research (Hamrick et al., 2022; Nam, 2023) processes social or market signals in isolation, never jointly. FinShield addresses both limitations.**



**This paper makes three novel contributions:**

**1. The first cross-modal social-market anomaly detection pipeline validated against real Indian NSE/BSE equities and SEBI enforcement orders**

**2. An empirical linguistic characterisation of Indian financial Telegram content, finding English dominance contrary to prior assumptions**

**3. A reproducible SEBI-backtest methodology for future Indian market surveillance research**



**---**



**## II. RELATED WORK**



**### A. Pump-and-Dump Detection**



**Hamrick et al. \[1] monitored cryptocurrency pump-and-dump events across Telegram and Discord over three years, detecting approximately 900 real P\&D events with F1 of 94.5% within 25 seconds of event start using real-time ML. Their work confirms that social coordination on Telegram precedes price anomalies — validating FinShield's theoretical basis. Nam \[2] combined Twitter and Telegram signals with Random Forest classification, achieving 85% sensitivity and 99% specificity on cryptocurrency markets. Both works are limited to cryptocurrency markets and English-language content.**



**### B. Hinglish / Code-Switched NLP**



**Garg et al. \[3] demonstrated that standard sentiment models produce misleading results on Hinglish tweets, advocating fine-tuned multilingual BERT models with subword tokenization for Indian code-switched text. Verma et al. \[4] compared fine-tuned multilingual BERT against BiLSTM baselines on Hinglish classification, showing that focal loss and balanced class weights address class imbalance — both applied in FinShield as balanced class weights \[Neutral: 3.14, OpenTip: 1.30, Suspicious: 0.52].**



**### C. MuRIL and Indian Language NLP**



**Google Research \[5] released MuRIL (Multilingual Representations for Indian Languages), a BERT-based transformer pre-trained on 17 Indian languages including Hindi, English, and code-switched text from Wikipedia and Common Crawl. MuRIL significantly outperforms mBERT on Indian language tasks. FinShield fine-tunes MuRIL on a domain-specific financial social media corpus.**



**### D. Research Gap**



**No published system applies cross-modal social-market fusion to Indian NSE/BSE equities. All existing P\&D detection research targets cryptocurrency markets and uses English-only NLP. FinShield is, to our knowledge, the first system to validate cross-modal financial misinformation detection against real Indian regulatory enforcement data.**



**---**



**## III. METHODOLOGY**



**### A. System Architecture**



**FinShield operates as a four-stage cross-modal pipeline:**

