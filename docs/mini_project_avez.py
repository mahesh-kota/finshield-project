import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 1. Importing Libraries and Dataset
# ==========================================
# Update this path to where you saved the Kaggle dataset
dataset_path = r'C:\Users\mahes\Downloads\archive (1).zip\WA_Fn-UseC_-Telco-Customer-Churn.csv' 
dataset = pd.read_csv(dataset_path)

print("--- First 5 rows of Dataset ---")
print(dataset.head())

print("\n--- Missing Values Check ---")
print(dataset.isnull().sum())

print("\n--- Descriptive Statistics ---")
print(dataset.describe())

# Plotting Churn Distribution
print("\n--- Plotting Churn Distribution ---")
print(dataset['Churn'].value_counts())
sns.countplot(x='Churn', data=dataset, palette='coolwarm')
plt.title('Churn Distribution')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# ==========================================
# 2. Data Preprocessing
# ==========================================
# Handling Missing and Incorrect Values in TotalCharges
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
dataset['TotalCharges'].fillna(dataset['TotalCharges'].median(), inplace=True)

# Handling Categorical Variables
labelencoder = LabelEncoder()
categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
    'PaperlessBilling', 'PaymentMethod', 'Churn'
]

for col in categorical_cols:
    dataset[col] = labelencoder.fit_transform(dataset[col])

# Feature Selection and Splitting Data
X = dataset.drop(['customerID', 'Churn'], axis=1)
y = dataset['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 3. Model Training and Prediction
# ==========================================
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# ==========================================
# 4. Model Evaluation
# ==========================================
# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap="coolwarm")
plt.title('Confusion Matrix')
plt.show()