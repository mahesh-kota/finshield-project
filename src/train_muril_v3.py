import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from datasets import Dataset

df = pd.read_csv('data/telegram_posts_labelled.csv')
df = df.dropna(subset=['text', 'label'])
df['label'] = df['label'].astype(int)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
print("Train:", train_df['label'].value_counts().sort_index().to_dict())
print("Val:  ", val_df['label'].value_counts().sort_index().to_dict())

counts = train_df['label'].value_counts().sort_index()
weights = (len(train_df) / (3 * counts)).values
print("Class weights:", dict(zip([0,1,2], weights.round(2))))

tok = AutoTokenizer.from_pretrained("google/muril-base-cased")
def tokenize(ex):
    return tok(ex['text'], truncation=True, padding='max_length', max_length=128)

train_ds = Dataset.from_pandas(train_df[['text','label']].reset_index(drop=True)).map(tokenize, batched=True)
val_ds   = Dataset.from_pandas(val_df[['text','label']].reset_index(drop=True)).map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("google/muril-base-cased", num_labels=3)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        loss = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float).to(out.logits.device)
        )(out.logits.view(-1, 3), labels.view(-1))
        return (loss, out) if return_outputs else loss

def metrics(ep):
    preds = np.argmax(ep.predictions, axis=-1)
    return {'macro_f1': f1_score(ep.label_ids, preds, average='macro')}

args = TrainingArguments(
    output_dir='./muril_v3',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1',
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    report_to='none',
)

trainer = WeightedTrainer(model=model, args=args,
                          train_dataset=train_ds, eval_dataset=val_ds,
                          compute_metrics=metrics)
trainer.train()

pred = trainer.predict(val_ds)
y_pred = np.argmax(pred.predictions, axis=-1)
y_true = pred.label_ids

print("\n" + classification_report(y_true, y_pred,
      target_names=['neutral','open_tip','suspicious'], digits=3))
print("Confusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_true, y_pred))

val_df = val_df.reset_index(drop=True)
val_df['pred'] = y_pred
bad = val_df[(val_df['label']==0) & (val_df['pred']!=0)]
print(f"\nMisclassified neutrals ({len(bad)}):")
for _, r in bad.iterrows():
    print(f"  true=0 pred={r['pred']}  {r['text'][:120]}")
