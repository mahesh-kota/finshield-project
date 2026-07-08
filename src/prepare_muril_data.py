import pandas as pd
from sklearn.model_selection import train_test_split
import json

df = pd.read_csv("data/telegram_posts_labelled.csv")

# Clean text - remove markdown asterisks, extra whitespace
df["text"] = df["text"].str.replace(r'\*+', '', regex=True).str.strip()
df["text"] = df["text"].str.replace(r'\n+', ' ', regex=True)

# Keep only text and label
df = df[["text", "label"]].dropna()

# Train/validation split (80/20)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

print(f"Training set: {len(train_df)} posts")
print(f"Validation set: {len(val_df)} posts")
print(f"\nTrain label distribution:\n{train_df['label'].value_counts()}")
print(f"\nVal label distribution:\n{val_df['label'].value_counts()}")

# Save as CSV for Colab
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
print("\nSaved: data/train.csv and data/val.csv")
