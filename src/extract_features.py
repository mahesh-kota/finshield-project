import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tools
print("Loading models...")
analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

MODEL_PATH = "models/muril_finshield_v2"
device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Load stock universe for ticker matching
with open("data/stock_universe.txt") as f:
    tickers = [line.strip().replace(".NS", "") for line in f if line.strip()]

# Urgency keywords common in Indian trading tip channels
URGENCY_KEYWORDS = [
    "now", "immediately", "urgent", "jackpot", "guaranteed",
    "sure shot", "100%", "free", "join", "limited", "today only",
    "abhi", "jaldi", "pakka", "confirmed", "zero to hero"
]

def get_muril_label(text):
    enc = tokenizer(str(text), padding="max_length", truncation=True,
                    max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()
    return pred, round(confidence, 3)

def extract_tickers(text):
    found = []
    text_upper = text.upper()
    for ticker in tickers:
        if ticker in text_upper:
            found.append(ticker)
    return found

def urgency_score(text):
    text_lower = text.lower()
    hits = sum(1 for kw in URGENCY_KEYWORDS if kw in text_lower)
    return min(hits / 3, 1.0)  # normalize to 0-1

def extract_price_target(text):
    pattern = r'(?:target|tgt|sl|stoploss)[:\s]*(\d+\.?\d*)'
    matches = re.findall(pattern, text.lower())
    return matches if matches else []

# Run on labelled dataset
df = pd.read_csv("data/telegram_posts_labelled.csv")
print(f"Processing {len(df)} posts...")

results = []
for idx, row in df.iterrows():
    text = str(row["text"])
    sentiment = analyzer.polarity_scores(text)
    muril_pred, muril_conf = get_muril_label(text)
    tickers_found = extract_tickers(text)
    urgency = urgency_score(text)
    prices = extract_price_target(text)

    results.append({
        "channel": row["channel"],
        "message_id": row["message_id"],
        "date": row["date"],
        "text": text[:100],
        "true_label": row["label"],
        "muril_pred": muril_pred,
        "muril_confidence": muril_conf,
        "sentiment_compound": sentiment["compound"],
        "sentiment_pos": sentiment["pos"],
        "sentiment_neg": sentiment["neg"],
        "urgency_score": urgency,
        "tickers_mentioned": str(tickers_found),
        "price_targets_found": str(prices)
    })

features_df = pd.DataFrame(results)
features_df.to_csv("data/features.csv", index=False)
print(f"Saved features to data/features.csv")
print(f"\nSample output:")
print(features_df[["muril_pred", "sentiment_compound", "urgency_score",
                     "tickers_mentioned"]].head(10))
print(f"\nTicker mentions found: {features_df['tickers_mentioned'].apply(lambda x: x != '[]').sum()} posts")
print(f"Average urgency score: {features_df['urgency_score'].mean():.3f}")
print(f"MuRIL accuracy on full set: {(features_df['muril_pred'] == features_df['true_label']).mean()*100:.1f}%")