import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

model = joblib.load("waterborne_disease_rf_model.pkl")

print("Saved Random Forest model loaded successfully!")


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")["disease"]

print("\nTest data loaded!")
print("Test shape:", X_test.shape)


# ============================================================
# 3. SAME FEATURE ENGINEERING - TEST DATA
# ============================================================

# Coliform ratio
X_test["coliform_ratio"] = (
    X_test["fecal_coliform_per_100ml"]
    / (X_test["total_coliform_per_100ml"] + 1)
)

# Log-transformed coliform ratio
X_test["log_coliform_ratio"] = np.log1p(
    X_test["coliform_ratio"]
)

# BOD / Dissolved Oxygen ratio
X_test["bod_do_ratio"] = (
    X_test["bod_mg_l"]
    / (X_test["dissolved_oxygen_mg_l"] + 1e-6)
)

# Sanitation risk
X_test["sanitation_risk"] = (
    X_test["open_defecation_rate"]
    * (1 - X_test["sewage_treatment_pct"] / 100)
)

# Remove temporary raw ratio
X_test = X_test.drop(columns=["coliform_ratio"])


# ============================================================
# 4. MAKE TEST PREDICTIONS
# ============================================================

print("\nMaking final test predictions...")

y_pred = model.predict(X_test)


# ============================================================
# 5. FINAL TEST ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n============================================================")
print("FINAL TEST RESULTS")
print("============================================================")

print("\nTest Accuracy:", accuracy)
print("Test Accuracy (%):", round(accuracy * 100, 2), "%")


# ============================================================
# 6. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=2
    )
)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 8. MACRO F1 SCORE
# ============================================================

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print("\nMacro F1 Score:", macro_f1)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("FINAL MODEL SUMMARY")
print("============================================================")

print("Model: Random Forest")
print("Trees: 100")
print("Features: 43")
print("Symptoms: INCLUDED")
print("Test Accuracy:", round(accuracy * 100, 2), "%")
print("Test Macro F1:", round(macro_f1, 4))

print("\nFinal test evaluation completed successfully!")