"""
Trains the Telco customer-churn model exactly as in the original notebook:
Decision Tree / Random Forest / XGBoost compared via 5-fold CV on SMOTE-balanced
data, Random Forest selected as the best performer, then saved as a pickle
artifact together with the label encoders.

Run this locally (`python train_model.py`) whenever scikit-learn / imbalanced-learn
gets upgraded, so the .pkl files always match the environment that unpickles them.
"""
import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["customerID"])

# TotalCharges has blank strings for brand-new customers -> treat as 0
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)

# Target encoding (explicit int cast so it isn't picked up as a categorical column below)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

# Encode every remaining categorical column and keep the encoders for inference
object_columns = df.drop(columns=["Churn"]).select_dtypes(include=["object", "string"]).columns
encoders = {}
for column in object_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Balancing training data with SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
}

print("\n5-fold cross-validation on SMOTE-balanced training data:")
for name, model in models.items():
    scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring="accuracy")
    print(f"  {name}: {np.mean(scores):.4f}")

print("\nTraining final Random Forest model...")
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_smote, y_train_smote)

y_pred = rfc.predict(X_test)
print("\nHold-out test performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

model_data = {"model": rfc, "features_names": X.columns.tolist()}
with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("\nSaved customer_churn_model.pkl and encoders.pkl")
