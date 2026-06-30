from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
uri = os.getenv("MONGO_URI")
client = MongoClient(uri)
db = client["finshield"]

# Clean up the test collection from Day 11
db.test_collection.drop()
print("Dropped test_collection")

# Collection 1: social_posts (Telegram, Reddit later)
# Structure: one document per scraped post
sample_post = {
    "platform": "telegram",
    "channel": "BankNifty_Nifty_Intraday_Sensex",
    "message_id": 3546,
    "date": "2026-06-29T07:49:26Z",
    "text": "sample post text",
    "sender_id": None,
    "tickers_mentioned": [],       # filled later by NER (Stage 2)
    "sentiment_score": None,       # filled later (Stage 2)
    "urgency_score": None,         # filled later (Stage 2)
    "processed": False
}
db.social_posts.insert_one(sample_post)
print("Created social_posts collection")

# Collection 2: market_ticks
# Structure: one document per stock per day
sample_tick = {
    "ticker": "RELIANCE.NS",
    "date": "2026-06-29",
    "open": 1280.5,
    "high": 1290.0,
    "low": 1275.0,
    "close": 1285.3,
    "volume": 5000000,
    "abnormal_return": None,       # filled later (Stage 3)
    "volume_zscore": None          # filled later (Stage 3)
}
db.market_ticks.insert_one(sample_tick)
print("Created market_ticks collection")

# Collection 3: alerts (final output, Stage 4)
sample_alert = {
    "ticker": "RELIANCE.NS",
    "date": "2026-06-29",
    "risk_score": None,
    "evidence_post_ids": [],
    "status": "draft"
}
db.alerts.insert_one(sample_alert)
print("Created alerts collection")

print("\nAll collections in database:", db.list_collection_names())
