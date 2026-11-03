import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

df = pd.read_csv('data/telegram_posts_labelled.csv').dropna(subset=['text','label'])
df['label'] = df['label'].astype(int)

X_tr, X_va, y_tr, y_va = train_test_split(
    df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=5000)),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000)),
])
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_va)

print(classification_report(y_va, y_pred,
      target_names=['neutral','open_tip','suspicious'], digits=3, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_va, y_pred))
