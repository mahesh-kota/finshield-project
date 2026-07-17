from pymongo import MongoClient
from dotenv import load_dotenv
import os
import certifi

load_dotenv()
uri = os.getenv("MONGO_URI")

client = MongoClient(uri, tlsCAFile=certifi.where())
db = client["finshield"]
print("Collections:", db.list_collection_names())
