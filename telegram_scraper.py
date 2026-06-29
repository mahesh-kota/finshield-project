from telethon.sync import TelegramClient
from dotenv import load_dotenv
import os
import csv

load_dotenv()

api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")

client = TelegramClient("finshield_session", api_id, api_hash)

# Add 2 public finance channel usernames here (without @)
channels = ["BankNifty_Nifty_Intraday_Sensex", "Tips_Zero_To_Hero_Trading", "best_intraday_stock", "intraday_stock_option_banknifty", "Nifty451"]

with client:
    all_messages = []
    for channel in channels:
        print(f"Fetching from: {channel}")
        try:
          for message in client.iter_messages(channel, limit=100):
                if message.text:
                    all_messages.append({
                        "channel": channel,
                        "message_id": message.id,
                        "date": message.date,
                        "text": message.text,
                        "sender_id": message.sender_id
                    })
        except Exception as e:
            print(f"Could not fetch from {channel}: {e}")

    print(f"\nTotal messages collected: {len(all_messages)}")

    # Save to CSV
    if all_messages:
        with open("data/telegram_posts.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_messages[0].keys())
            writer.writeheader()
            writer.writerows(all_messages)
        print("Saved to data/telegram_posts.csv")
