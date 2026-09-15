import pandas as pd
import numpy as np

# Load training and validation data
X_train = pd.read_csv("X_train.csv")
X_val = pd.read_csv("X_val.csv")

# ---------- Feature Engineering ----------

# 1. Coliform ratio
X_train["coliform_ratio"] = (
    X_train["fecal_coliform_per_100ml"] /
    (X_train["total_coliform_per_100ml"] + 1)
)

X_val["coliform_ratio"] = (
    X_val["fecal_coliform_per_100ml"] /
    (X_val["total_coliform_per_100ml"] + 1)
)
X_test = pd.read_csv("X_test.csv")

X_test["coliform_ratio"] = (
    X_test["fecal_coliform_per_100ml"] /
    (X_test["total_coliform_per_100ml"] + 1)
)

X_test["log_coliform_ratio"] = np.log1p(X_test["coliform_ratio"])

X_test["bod_do_ratio"] = (
    X_test["bod_mg_l"] /
    (X_test["dissolved_oxygen_mg_l"] + 1e-6)
)

X_test["sanitation_risk"] = (
    X_test["open_defecation_rate"] *
    (1 - X_test["sewage_treatment_pct"] / 100)
)

print("Test shape:", X_test.shape)

# 2. Log coliform ratio
X_train["log_coliform_ratio"] = np.log1p(X_train["coliform_ratio"])
X_val["log_coliform_ratio"] = np.log1p(X_val["coliform_ratio"])

# 3. BOD / DO ratio
X_train["bod_do_ratio"] = (
    X_train["bod_mg_l"] /
    (X_train["dissolved_oxygen_mg_l"] + 1e-6)
)

X_val["bod_do_ratio"] = (
    X_val["bod_mg_l"] /
    (X_val["dissolved_oxygen_mg_l"] + 1e-6)
)

# 4. Sanitation risk
X_train["sanitation_risk"] = (
    X_train["open_defecation_rate"] *
    (1 - X_train["sewage_treatment_pct"] / 100)
)

X_val["sanitation_risk"] = (
    X_val["open_defecation_rate"] *
    (1 - X_val["sewage_treatment_pct"] / 100)
)

# ---------- Check ----------

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

print("\nNew features:")
print("log_coliform_ratio")
print("bod_do_ratio")
print("sanitation_risk")
y_train = pd.read_csv("y_train.csv")

print("\nAverage engineered feature values by disease class:")

print(
    X_train.groupby(y_train["disease"])[
        ["log_coliform_ratio", "bod_do_ratio", "sanitation_risk"]
    ].mean()
)
X_train = X_train.drop(columns=["coliform_ratio"])
X_val = X_val.drop(columns=["coliform_ratio"])
X_test = X_test.drop(columns=["coliform_ratio"])
print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)
