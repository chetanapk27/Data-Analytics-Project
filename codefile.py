import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# 1. Load the optimized dataset
print("Loading dataset from Paysim_small.csv...")
df = pd.read_csv("Paysim_small.csv")

# 2. Feature Engineering
print("Engineering features...")
df_clean = df.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis=1)

# Balance error calculations
df_clean["errorBalanceOrg"] = (
    df_clean["oldbalanceOrg"] - df_clean["amount"] - df_clean["newbalanceOrig"]
)
df_clean["errorBalanceDest"] = (
    df_clean["newbalanceDest"] + df_clean["amount"] - df_clean["oldbalanceDest"]
)

# One-hot encode transaction types
df_clean = pd.get_dummies(df_clean, columns=["type"], drop_first=True)

# 3. Train-Test Split
X = df_clean.drop("isFraud", axis=1)
y = df_clean["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Model Training
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=10, class_weight="balanced", random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# 5. Evaluation Metrics
y_pred = model.predict(X_test)
print("\n--- Model Performance Report ---")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# 6. Save Model and Columns for API/UI
joblib.dump(model, "fraud_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")
print("Model successfully trained, verified, and saved!")