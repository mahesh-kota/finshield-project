import pandas as pd
import re

df = pd.read_csv("data/telegram_posts_labelled.csv")

# Simple check: does the text contain any Devanagari script characters?
devanagari_pattern = re.compile(r'[\u0900-\u097F]')

has_devanagari = df["text"].apply(lambda x: bool(devanagari_pattern.search(str(x))))
print(f"Posts with Devanagari script: {has_devanagari.sum()} out of {len(df)}")

# Also check common Romanized Hindi trading words
hindi_words = ["lagao", "le lo", "le lijiye", "achha", "theek", "kharido", "becho", "kal", "aaj", "lagega"]
pattern = "|".join(hindi_words)
has_romanized = df["text"].str.lower().str.contains(pattern, regex=True, na=False)
print(f"Posts with common Romanized Hindi words: {has_romanized.sum()} out of {len(df)}")
