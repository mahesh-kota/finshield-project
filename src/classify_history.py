"""
FinShield - Classify scraped history with MuRIL.
Writes data/history_predictions.csv. These are model predictions,
not manual labels - do not mix them into training data.
"""
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

v3 = "models/muril_finshield_v3"
path = v3 if os.path.exists(v3 + "/model.safetensors") else "models/muril_finshield_v2"
print("Using model:", path)
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path).eval()

df = pd.read_csv("data/telegram_history.csv")
texts = (df["text"].astype(str)
         .str.replace(r"\*+", "", regex=True)
         .str.replace(r"\n+", " ", regex=True).tolist())

preds, confs = [], []
B = 32
with torch.no_grad():
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i + B], padding=True, truncation=True,
                  max_length=128, return_tensors="pt")
        probs = torch.softmax(model(**enc).logits, dim=-1)
        c, p = probs.max(dim=-1)
        preds += p.tolist()
        confs += c.tolist()
        if (i // B) % 10 == 0:
            print(f"{i}/{len(texts)}")

df["pred"] = preds
df["conf"] = confs
df.to_csv("data/history_predictions.csv", index=False)
print(df["pred"].value_counts())