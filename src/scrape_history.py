"""
FinShield - Full-History Telegram Scraper
Pulls up to MAX_PER_CHANNEL messages (newest first) from high-volume
public channels into data/telegram_history.csv. Kept separate from
telegram_posts.csv so labelled data is untouched.
"""
from telethon.sync import TelegramClient
from dotenv import load_dotenv
import pandas as pd
import os
import csv

load_dotenv()
api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")

CHANNELS = [
    "best_intraday_stock",
    "Nifty451",
    "intraday_stock_option_banknifty",
    "BankNifty_Nifty_Intraday_Sensex",
]
MAX_PER_CHANNEL = 5000
MIN_LEN = 15

rows, seen = [], set()
with TelegramClient("finshield_session", api_id, api_hash) as client:
    for ch in CHANNELS:
        kept = scanned = 0
        try:
            for m in client.iter_messages(ch, limit=MAX_PER_CHANNEL):
                scanned += 1
                if scanned % 500 == 0:
                    print(f"  {ch}: scanned {scanned}, kept {kept}")
                if not m.text:
                    continue
                text = m.text.strip()
                if len(text) < MIN_LEN or text in seen:
                    continue
                seen.add(text)
                rows.append({"channel": ch, "message_id": m.id,
                             "date": m.date, "text": text})
                kept += 1
        except Exception as e:
            print(f"Could not fetch {ch}: {e}")
        print(f"{ch}: scanned {scanned}, kept {kept}")

with open("data/telegram_history.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["channel", "message_id", "date", "text"])
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved {len(rows)} messages to data/telegram_history.csv")

df = pd.read_csv("data/telegram_history.csv")
df["day"] = pd.to_datetime(df["date"], utc=True).dt.date
print(df.groupby("channel").agg(posts=("text", "count"), days=("day", "nunique"),
                                first=("day", "min"), last=("day", "max")))
print("Distinct post days overall:", df["day"].nunique())