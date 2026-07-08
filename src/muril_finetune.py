# FinShield - MuRIL Fine-tuning Script
# Run this on Google Colab (Runtime -> Change runtime type -> T4 GPU)

# Step 1: Install dependencies
# !pip install transformers datasets scikit-learn pandas -q

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score

# Step 2: Load data
# Upload train.csv and val.csv to Colab first (Files panel on left)
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")

print(f"Train: {len(train_df)}, Val: {len(val_df)}")
print(train_df["label"].value_counts())

# Step 3: Load MuRIL tokenizer and model
MODEL_NAME = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

# Step 4: Tokenize
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Step 5: Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"weighted_f1": f1}

# Step 6: Training arguments
training_args = TrainingArguments(
    output_dir="./muril_finshield",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="weighted_f1",
    logging_steps=10,
)

# Step 7: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Step 8: Train
print("Starting MuRIL fine-tuning...")
trainer.train()

# Step 9: Evaluate
results = trainer.evaluate()
print("\nFinal evaluation results:")
print(results)

# Step 10: Save model
trainer.save_model("./muril_finshield_final")
tokenizer.save_pretrained("./muril_finshield_final")
print("\nModel saved to ./muril_finshield_final")

# Step 11: Detailed classification report
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids
print("\nClassification Report:")
print(classification_report(labels, preds,
      target_names=["Neutral(0)", "OpenTip(1)", "Suspicious(2)"]))
