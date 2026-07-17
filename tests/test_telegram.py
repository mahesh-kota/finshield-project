from telethon.sync import TelegramClient
from dotenv import load_dotenv
import os

load_dotenv()

api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")

client = TelegramClient("finshield_session", api_id, api_hash)

with client:
    me = client.get_me()
    print("Connected successfully as:", me.first_name)
