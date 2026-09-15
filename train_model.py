import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")["disease"]

print("Training data loaded!")
print("Training shape:", X_train.shape)


# ============================================================
# 2. FEATURE ENGINEERING - TRAINING DATA
# ============================================================

# Coliform ratio
X_train["coliform_ratio"] = (
    X_train["fecal_coliform_per_100ml"]
    / (X_train["total_coliform_per_100ml"] + 1)
)

# Log-transformed coliform ratio
X_train["log_coliform_ratio"] = np.log1p(
    X_train["coliform_ratio"]
)

# BOD / Dissolved Oxygen ratio
X_train["bod_do_ratio"] = (
    X_train["bod_mg_l"]
    / (X_train["dissolved_oxygen_mg_l"] + 1e-6)
)

# Sanitation risk
X_train["sanitation_risk"] = (
    X_train["open_defecation_rate"]
    * (1 - X_train["sewage_treatment_pct"] / 100)
)

# Remove temporary raw ratio
X_train = X_train.drop(columns=["coliform_ratio"])


# ============================================================
# IMPORTANT:
# SYMPTOM FEATURES ARE KEPT
# ============================================================

print("\nSymptoms are INCLUDED in the model.")


# ============================================================
# 3. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Random Forest training completed!")


# ============================================================
# 4. SAVE TRAINED MODEL
# ============================================================

joblib.dump(model, "waterborne_disease_rf_model.pkl")

print("Model saved as:")
print("waterborne_disease_rf_model.pkl")


# ============================================================
# 5. LOAD VALIDATION DATA
# ============================================================

X_val = pd.read_csv("X_val.csv")
y_val = pd.read_csv("y_val.csv")["disease"]

print("\nValidation data loaded!")
print("Validation shape:", X_val.shape)


# ============================================================
# 6. SAME FEATURE ENGINEERING - VALIDATION DATA
# ============================================================

# Coliform ratio
X_val["coliform_ratio"] = (
    X_val["fecal_coliform_per_100ml"]
    / (X_val["total_coliform_per_100ml"] + 1)
)

# Log-transformed coliform ratio
X_val["log_coliform_ratio"] = np.log1p(
    X_val["coliform_ratio"]
)

# BOD / Dissolved Oxygen ratio
X_val["bod_do_ratio"] = (
    X_val["bod_mg_l"]
    / (X_val["dissolved_oxygen_mg_l"] + 1e-6)
)

# Sanitation risk
X_val["sanitation_risk"] = (
    X_val["open_defecation_rate"]
    * (1 - X_val["sewage_treatment_pct"] / 100)
)

# Remove temporary raw ratio
X_val = X_val.drop(columns=["coliform_ratio"])


# ============================================================
# 7. MAKE VALIDATION PREDICTIONS
# ============================================================

print("\nMaking validation predictions...")

y_pred = model.predict(X_val)


# ============================================================
# 8. VALIDATION ACCURACY
# ============================================================

accuracy = accuracy_score(y_val, y_pred)

print("\n============================================================")
print("VALIDATION RESULTS")
print("============================================================")

print("\nValidation Accuracy:", accuracy)


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_pred,
        digits=2
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_val, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 11. MACRO F1 SCORE
# ============================================================

macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro"
)

print("\nMacro F1 Score:", macro_f1)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
)

print("\nTop 15 Important Features:")

print(
    importance
    .sort_values(ascending=False)
    .head(15)
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("MODEL SUMMARY")
print("============================================================")

print("Model: Random Forest")
print("Trees:", 100)
print("Features used:", X_train.shape[1])
print("Symptoms: INCLUDED")
print("Validation Accuracy:", round(accuracy * 100, 2), "%")
print("Macro F1 Score:", round(macro_f1, 4))

print("\nTraining and validation completed successfully!")