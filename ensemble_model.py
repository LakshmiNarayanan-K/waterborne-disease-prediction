import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# ==============================
# LOAD MODELS
# ==============================

print("Loading models...")

rf_model = joblib.load("waterborne_disease_rf_model.pkl")
xgb_model = joblib.load("waterborne_disease_xgboost_model.pkl")


# ==============================
# LOAD VALIDATION DATA
# ==============================

X_val = pd.read_csv("X_val.csv")
y_val = pd.read_csv("y_val.csv")


# ==============================
# FEATURE ENGINEERING
# ==============================

def add_features(df):

    df = df.copy()

    df["log_coliform_ratio"] = np.log1p(
        df["fecal_coliform_per_100ml"] /
        (df["total_coliform_per_100ml"] + 1)
    )

    df["bod_do_ratio"] = (
        df["bod_mg_l"] /
        (df["dissolved_oxygen_mg_l"] + 1e-6)
    )

    df["sanitation_risk"] = (
        df["open_defecation_rate"] *
        (1 - df["sewage_treatment_pct"] / 100)
    )

    return df


X_val = add_features(X_val)


# ==============================
# CHECK FEATURES
# ==============================

print("Validation data shape:", X_val.shape)

print("RF features:", len(rf_model.feature_names_in_))
print("XGBoost features:", len(xgb_model.feature_names_in_))


# ==============================
# GET PROBABILITIES
# ==============================

print("\nGetting predictions...")

rf_proba = rf_model.predict_proba(X_val)

xgb_proba = xgb_model.predict_proba(X_val)


# ==============================
# SOFT VOTING ENSEMBLE
# ==============================

# Equal weight
ensemble_proba = (
    0.5 * rf_proba +
    0.5 * xgb_proba
)

ensemble_pred = np.argmax(
    ensemble_proba,
    axis=1
)


# ==============================
# EVALUATION
# ==============================

accuracy = accuracy_score(
    y_val,
    ensemble_pred
)

macro_f1 = f1_score(
    y_val,
    ensemble_pred,
    average="macro"
)


print("\n================================")
print("RANDOM FOREST + XGBOOST ENSEMBLE")
print("================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")


# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        ensemble_pred
    )
)


# ==============================
# CONFUSION MATRIX
# ==============================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_val,
        ensemble_pred
    )
)


# ==============================
# SAVE ENSEMBLE INFORMATION
# ==============================

ensemble_info = {
    "rf_weight": 0.5,
    "xgb_weight": 0.5,
    "disease_classes": {
        0: "Cholera",
        1: "Dysentery",
        2: "Giardiasis",
        3: "Hepatitis_A",
        4: "Hepatitis_E",
        5: "Leptospirosis",
        6: "No_Disease",
        7: "Typhoid"
    }
}

joblib.dump(
    ensemble_info,
    "waterborne_disease_ensemble_info.pkl"
)

print("\nEnsemble evaluation completed!")