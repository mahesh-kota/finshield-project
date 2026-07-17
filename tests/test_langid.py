from langdetect import detect, DetectorFactory
import pandas as pd

DetectorFactory.seed = 0  # for consistent results

df = pd.read_csv("data/telegram_posts_labelled.csv")

print("Testing language detection on 10 sample posts:\n")

for idx, row in df.sample(10, random_state=1).iterrows():
    text = row["text"]
    try:
        lang = detect(text)
    except Exception:
        lang = "unknown"
    print(f"[{lang}] {text[:80]}")
    print()
