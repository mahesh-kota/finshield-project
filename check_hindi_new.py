import pandas as pd
import re

df = pd.read_csv("data/telegram_posts.csv")
print(f"Total posts in file: {len(df)}")

devanagari_pattern = re.compile(r'[\u0900-\u097F]')
has_devanagari = df["text"].apply(lambda x: bool(devanagari_pattern.search(str(x))))
print(f"Posts with Devanagari script: {has_devanagari.sum()} out of {len(df)}")

hindi_words = ["lagao", "le lo", "le lijiye", "achha", "theek", "kharido", "becho", "kal", "aaj", "lagega", "karo", "hoga", "raha"]
pattern = "|".join(hindi_words)
has_romanized = df["text"].str.lower().str.contains(pattern, regex=True, na=False)
print(f"Posts with common Romanized Hindi words: {has_romanized.sum()} out of {len(df)}")

# Show a few samples that have Hindi content
print("\nSample Devanagari posts:")
for text in df[has_devanagari]["text"].head(3):
    print("-", text[:150])

print("\nSample Romanized Hindi posts:")
for text in df[has_romanized]["text"].head(3):
    print("-", text[:150])
