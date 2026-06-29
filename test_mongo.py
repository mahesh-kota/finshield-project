from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
uri = os.getenv("MONGO_URI")

client = MongoClient(uri)
db = client["finshield"]

db.test_collection.insert_one({"status": "connected", "project": "FinShield"})
result = db.test_collection.find_one({"project": "FinShield"})
print("MongoDB connected successfully:", result)
