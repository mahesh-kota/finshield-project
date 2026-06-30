from telethon.sync import TelegramClient
from dotenv import load_dotenv
import os
import csv

load_dotenv()

api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")

client = TelegramClient("finshield_session", api_id, api_hash)

channels = ["intraday1341", "Share_market_Intraday_tips", "intradaytradingchart", "AmitBhawaniOfficial02"]

MIN_TEXT_LENGTH = 15  # skip very short/junk messages

with client:
    all_messages = []
    seen_texts = set()  # to skip exact duplicate forwards

    for channel in channels:
        print(f"Fetching from: {channel}")
        try:
            count = 0
            for message in client.iter_messages(channel, limit=200):
                if not message.text:
                    continue
                text = message.text.strip()
                if len(text) < MIN_TEXT_LENGTH:
                    continue
                if text in seen_texts:
                    continue
                seen_texts.add(text)

                all_messages.append({
                    "channel": channel,
                    "message_id": message.id,
                    "date": message.date,
                    "text": text,
                    "sender_id": message.sender_id
                })
                count += 1
            print(f"  -> collected {count} clean messages")
        except Exception as e:
            print(f"Could not fetch from {channel}: {e}")

    print(f"\nTotal clean messages collected: {len(all_messages)}")

    if all_messages:
        with open("data/telegram_posts.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_messages[0].keys())
            writer.writeheader()
            writer.writerows(all_messages)
        print("Saved to data/telegram_posts.csv")