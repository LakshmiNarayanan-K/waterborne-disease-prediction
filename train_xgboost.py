import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ==============================
# 1. Load training and validation data
# ==============================

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

X_val = pd.read_csv("X_val.csv")
y_val = pd.read_csv("y_val.csv")


# ==============================
# 2. Feature Engineering
# ==============================

def add_features(df):

    df = df.copy()

    # Coliform ratio
    df["log_coliform_ratio"] = __import__("numpy").log1p(
        df["fecal_coliform_per_100ml"] /
        (df["total_coliform_per_100ml"] + 1)
    )

    # BOD / DO ratio
    df["bod_do_ratio"] = (
        df["bod_mg_l"] /
        (df["dissolved_oxygen_mg_l"] + 1e-6)
    )

    # Sanitation risk
    df["sanitation_risk"] = (
        df["open_defecation_rate"] *
        (1 - df["sewage_treatment_pct"] / 100)
    )

    return df


X_train = add_features(X_train)
X_val = add_features(X_val)


# ==============================
# 3. Train XGBoost
# ==============================

print("Training XGBoost model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=8,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train.values.ravel())


# ==============================
# 4. Validation Prediction
# ==============================

y_pred = model.predict(X_val)


# ==============================
# 5. Evaluation
# ==============================

accuracy = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(y_val, y_pred, average="macro")

print("\n==============================")
print("XGBOOST VALIDATION RESULTS")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred))


# ==============================
# 6. Save Model
# ==============================

joblib.dump(model, "waterborne_disease_xgboost_model.pkl")

print("\nXGBoost model saved successfully!")