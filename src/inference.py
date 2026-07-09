import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/muril_finshield_v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model = model.to(device)
model.eval()

LABELS = {0: "Neutral", 1: "OpenTip", 2: "Suspicious"}

def predict(text):
    enc = tokenizer(str(text), padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item()
    return LABELS[pred], round(confidence, 3), probs[0].tolist()

df = pd.read_csv("data/val.csv")
label_map = {0: "Neutral", 1: "OpenTip", 2: "Suspicious"}

print("\n=== VALIDATION SET PREDICTIONS ===\n")
correct = 0
errors = []

for idx, row in df.iterrows():
    pred_label, confidence, probs = predict(row["text"])
    true_label = label_map[row["label"]]
    is_correct = pred_label == true_label
    if is_correct:
        correct += 1
    else:
        errors.append({"true": true_label, "pred": pred_label, "confidence": confidence, "text": row["text"][:100]})

print(f"Accuracy: {correct}/{len(df)} = {correct/len(df)*100:.1f}%")
print(f"Errors: {len(errors)}\n")

print("=== ERROR ANALYSIS ===")
for i, err in enumerate(errors):
    print(f"\nError {i+1}:")
    print(f"  True: {err['true']} | Predicted: {err['pred']} | Confidence: {err['confidence']}")
    print(f"  Text: {err['text']}...")

print("\n=== CUSTOM TEST POSTS ===\n")
test_posts = [
    "NIFTY data positive. BANKNIFTY neutral. VIX 2.3% up. Keep booking profits.",
    "🔥 intraday buy Call 🔥\nRELIANCE 1240-1245\nTarget 🎯 1260\nSl 1221",
    "JOIN NOW!! t.me/+abc123 ZERO TO HERO CALLS Premium Tips FREE!!",
    "ITI 235 to 290 Almost 30% returns in 20 days ✅✅🔥🔥",
    "Kal ka market kaisa rahega? BANKNIFTY upar jayega kya?"
]

for post in test_posts:
    pred, conf, _ = predict(post)
    print(f"[{pred} | {conf:.0%}] {post[:80]}")