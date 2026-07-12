**# Week 1 Summary (22 June - 28 June 2026)**



**## Completed**

**- Day 1: Project setup - GitHub repo, folder structure, venv created**

**- Day 2: Literature review - Hinglish NLP papers (mBERT, MuRIL, focal loss for imbalance)**

**- Day 3: Literature review - pump-and-dump detection papers (XGBoost, Random Forest, Telegram/Twitter signal fusion)**

**- Day 4: Tool research - NetworkX, Streamlit, FastAPI all installed and verified working**

**- Day 5: NSE/BSE data access - yfinance pull tested successfully on RELIANCE.NS (101 rows)**

**- Day 6: Telegram API setup - Telethon connected and authenticated successfully**

**- Day 7: Weekly review (today)**



**## Key learnings this week**

**- MuRIL is the right model choice for Hinglish text - confirmed by literature**

**- Coordination signals on Telegram/Twitter precede price manipulation in existing crypto research - supports our Granger causality hypothesis**

**- Almost no existing research applies pump-and-dump detection to Indian equities specifically - this is our research gap**



**## Issues faced and resolved**

**- Git Bash bracketed-paste mode kept corrupting multi-line pasted commands - fixed by typing manually or disabling bracketed paste**

**- venv creation was interrupted once, had to delete and recreate**

**- Telegram login hit a temporary rate limit ("too many tries") - resolved by waiting**

**- Accidentally ran git commands from home directory instead of project folder - cleaned up untracked files**



**## Open questions for guide**

**- Should we benchmark against crypto pump-and-dump papers even though our domain is equities?**

**- Is scraping public Telegram channels for research acceptable under our institute's ethics policy?**



**## Next week plan**

**- Twitter/X API research, MongoDB setup, yfinance scaling, Telegram scraper v1-v2, stock universe selection, database schema design**



**# Week 2 Summary (29 June - 05 July 2026)**



**## Completed**

**- Day 8: Twitter/X API research - found free tier too restrictive, snscrape broken on Python 3.13, decided to drop Twitter as a data source**

**- Day 9: Telegram scraper v1 - found 5 active public channels, 411 raw messages collected**

**- Day 10: Stock universe - 50 NSE mid/small-cap stocks selected and verified via yfinance**

**- Day 11: MongoDB Atlas setup - cluster created, connection tested successfully**

**- Day 12: MongoDB connection failed on retry (SSL handshake errors, likely network/firewall issue) - decided to drop MongoDB, use CSV/file storage instead. Also improved Telegram scraper with deduplication and minimum-length filtering (174 clean messages from 411 raw)**



**## Key learnings this week**

**- Official APIs (Twitter, Reddit) have heavy academic approval friction - public scraping (Telegram) is more reliable for a student project timeline**

**- Large channel subscriber counts don't always mean high text volume - some channels post mostly images/charts**

**- Tip messages follow a consistent template pattern (emoji + "Buy Call" + ticker + target) - this could become a useful feature for misinformation detection later**

**- Infrastructure decisions (like MongoDB vs CSV) should be driven by actual data scale, not assumed "best practice" - CSV is the right choice for this project's size**



**## Issues faced and resolved**

**- snscrape incompatible with Python 3.13 (uses deprecated import method) - confirmed not usable, dropped in favor of skipping Twitter entirely**

**- Reddit API requires university email and formal research proposal - submitted, pending approval**

**- MongoDB Atlas SSL handshake failures on second connection attempt - tried certifi upgrade and explicit TLS CA file, both failed identically across all 3 shards, pointing to network-level block rather than code issue - decided to stop debugging and use local CSV storage instead**



**## Open questions for guide**

**- Should we benchmark against crypto pump-and-dump papers even though our domain is equities? (carried over from Week 1)**

**- Is scraping public Telegram channels for research acceptable under our institute's ethics policy? (carried over from Week 1)**

**- Is it acceptable to exclude Twitter/X from our data sources given current API restrictions, and document this as a scoping decision?**



**## Next week plan**

**- Find more active Telegram channels to grow dataset further**

**- Check on Reddit API approval status**

**- Begin manual annotation of collected posts (misinformation / not) to prepare for MuRIL fine-tuning**

**- Start Hinglish language ID and normalisation pipeline (Stage 2 begins)**



**## Research finding (Day 15)**

**Tested both large broadcast-style Telegram channels and smaller Hindi-named discussion groups for Hinglish content. Result: <1% of sampled posts contained genuine Hindi-English code-switching; English trading jargon (CE/PE/SL/Target) dominates regardless of channel orientation. Decision: proceed with MuRIL fine-tuning on existing English-dominant dataset, reframe NLP contribution as "robust to Indian trading community linguistic patterns" rather than "Hinglish-specific." This is documented as an empirical finding for the paper rather than a project limitation.**



**# Week 3 Summary (07 July - 13 July 2026)**



**## Completed**

**- Day 16-18: MuRIL fine-tuning v1 and v2 - final weighted F1=0.92, accuracy=91.4%**

**- Day 19: Per-post feature extraction - VADER sentiment, urgency score, ticker NER**

**- Day 20: Market anomaly detection - 47 stocks, 50,101 rows, 2304 anomaly days (Jan 2022-Jul 2026)**

**- Day 21: Granger causality test - confirmed social signals lead market by 1-2 days (p=0.004)**

**- Day 22: Isolation Forest cross-modal fusion - 8 anomaly days flagged from 73 overlapping days**

**- Day 23: Streamlit dashboard complete - price charts, anomaly heatmap, MuRIL pie chart, flagged posts table**

**- Day 24: SEBI backtest - 75% recall (3/4 cases), Precision@8=0.38, 11-day early detection**



**## Key results this week**

**- Granger causality p=0.004 at lag 1 (after first-differencing) — strongest quantitative finding**

**- SEBI backtest confirms system detects real enforcement cases 11 days before regulatory action**

**- Full pipeline operational end-to-end: scrape → classify → anomaly → dashboard**



**## Issues faced**

**- Isolation Forest requires date overlap between social and market data — resolved by extending market data to Jan 2022**

**- Granger test required first-differencing to achieve stationarity — corrected, finding holds**

**- SEBI backtest stocks not in our universe caused 1 missed case — documented as scope limitation**



**## Status: All 4 pipeline stages complete**

**Stage 1 (Data) ✅ | Stage 2 (NLP) ✅ | Stage 3 (Anomaly) ✅ | Stage 4 (Dashboard+Eval) ✅**

