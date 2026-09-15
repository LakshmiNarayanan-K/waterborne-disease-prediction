import pandas as pd
import numpy as np
import joblib


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

model = joblib.load("waterborne_disease_rf_model.pkl")


# ============================================================
# 2. DISEASE CLASS MAPPING
# ============================================================

# Verified by matching X_train with cleaned_dataset.csv

disease_mapping = {
    0: "Cholera",
    1: "Dysentery",
    2: "Giardiasis",
    3: "Hepatitis_A",
    4: "Hepatitis_E",
    5: "Leptospirosis",
    6: "No_Disease",
    7: "Typhoid"
}


# ============================================================
# 3. SYSTEM HEADER
# ============================================================

print("\n============================================================")
print("          WATERBORNE DISEASE PREDICTION SYSTEM")
print("============================================================")

print("\nRandom Forest model loaded successfully!")
print("Model:", type(model).__name__)
print("Number of features:", len(model.feature_names_in_))


# ============================================================
# 4. INPUT FUNCTIONS
# ============================================================

def get_int(prompt, min_value=None, max_value=None):

    while True:

        try:
            value = int(input(prompt))

            if min_value is not None and value < min_value:
                print(f"Please enter a value >= {min_value}.")
                continue

            if max_value is not None and value > max_value:
                print(f"Please enter a value <= {max_value}.")
                continue

            return value

        except ValueError:
            print("Please enter a valid integer.")


def get_float(prompt, min_value=None, max_value=None):

    while True:

        try:
            value = float(input(prompt))

            if min_value is not None and value < min_value:
                print(f"Please enter a value >= {min_value}.")
                continue

            if max_value is not None and value > max_value:
                print(f"Please enter a value <= {max_value}.")
                continue

            return value

        except ValueError:
            print("Please enter a valid number.")


# ============================================================
# 5. USER INPUT
# ============================================================

print("\n============================================================")
print("                    ENTER INPUT DATA")
print("============================================================")


# ------------------------------------------------------------
# LOCATION & DEMOGRAPHIC INFORMATION
# ------------------------------------------------------------

print("\n--- LOCATION & DEMOGRAPHIC INFORMATION ---")

state = get_int(
    "State code (0-7): ",
    0,
    7
)

district = get_int(
    "District code (0-83): ",
    0,
    83
)

region = get_int(
    "Region code (0): ",
    0,
    0
)

latitude = get_float(
    "Latitude: "
)

longitude = get_float(
    "Longitude: "
)

is_urban = get_int(
    "Urban/Rural (0=Rural, 1=Urban): ",
    0,
    1
)

population_density = get_float(
    "Population density: ",
    0
)

age = get_float(
    "Age: ",
    0,
    120
)

gender = get_int(
    "Gender code (0-1): ",
    0,
    1
)


# ------------------------------------------------------------
# WATER INFORMATION
# ------------------------------------------------------------

print("\n--- WATER INFORMATION ---")

water_source = get_int(
    "Water source code (0-6): ",
    0,
    6
)

water_treatment = get_int(
    "Water treatment code (0-3): ",
    0,
    3
)

water_quality_index = get_float(
    "Water Quality Index: "
)

ph = get_float(
    "pH: "
)

turbidity_ntu = get_float(
    "Turbidity (NTU): ",
    0
)

dissolved_oxygen_mg_l = get_float(
    "Dissolved Oxygen (mg/L): ",
    0
)

bod_mg_l = get_float(
    "BOD (mg/L): ",
    0
)

fecal_coliform_per_100ml = get_float(
    "Fecal Coliform (per 100 mL): ",
    0
)

total_coliform_per_100ml = get_float(
    "Total Coliform (per 100 mL): ",
    0
)

tds_mg_l = get_float(
    "TDS (mg/L): ",
    0
)

nitrate_mg_l = get_float(
    "Nitrate (mg/L): ",
    0
)

fluoride_mg_l = get_float(
    "Fluoride (mg/L): ",
    0
)

arsenic_ug_l = get_float(
    "Arsenic (µg/L): ",
    0
)


# ------------------------------------------------------------
# SANITATION & HYGIENE
# ------------------------------------------------------------

print("\n--- SANITATION & HYGIENE ---")

open_defecation_rate = get_float(
    "Open defecation rate (%): ",
    0,
    100
)

toilet_access = get_float(
    "Toilet access (%): ",
    0,
    100
)

sewage_treatment_pct = get_float(
    "Sewage treatment (%): ",
    0,
    100
)

handwashing_practice = get_int(
    "Handwashing practice code (0-2): ",
    0,
    2
)


# ------------------------------------------------------------
# CLIMATE INFORMATION
# ------------------------------------------------------------

print("\n--- CLIMATE INFORMATION ---")

month = get_int(
    "Month (1-12): ",
    1,
    12
)

season = get_int(
    "Season code (0-3): ",
    0,
    3
)

avg_temperature_c = get_float(
    "Average temperature (°C): "
)

avg_rainfall_mm = get_float(
    "Average rainfall (mm): ",
    0
)

avg_humidity_pct = get_float(
    "Average humidity (%): ",
    0,
    100
)

flooding = get_int(
    "Flooding (0=No, 1=Yes): ",
    0,
    1
)


# ------------------------------------------------------------
# SYMPTOMS
# ------------------------------------------------------------

print("\n--- SYMPTOMS ---")
print("Enter 0 = No, 1 = Yes")

symptom_diarrhea = get_int(
    "Diarrhea: ",
    0,
    1
)

symptom_vomiting = get_int(
    "Vomiting: ",
    0,
    1
)

symptom_fever = get_int(
    "Fever: ",
    0,
    1
)

symptom_abdominal_pain = get_int(
    "Abdominal pain: ",
    0,
    1
)

symptom_dehydration = get_int(
    "Dehydration: ",
    0,
    1
)

symptom_jaundice = get_int(
    "Jaundice: ",
    0,
    1
)

symptom_bloody_stool = get_int(
    "Bloody stool: ",
    0,
    1
)

symptom_skin_rash = get_int(
    "Skin rash: ",
    0,
    1
)


# ============================================================
# 6. CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame([{

    "state": state,
    "district": district,
    "region": region,

    "latitude": latitude,
    "longitude": longitude,

    "is_urban": is_urban,
    "population_density": population_density,
    "age": age,
    "gender": gender,

    "water_source": water_source,
    "water_treatment": water_treatment,

    "water_quality_index": water_quality_index,
    "ph": ph,
    "turbidity_ntu": turbidity_ntu,

    "dissolved_oxygen_mg_l": dissolved_oxygen_mg_l,
    "bod_mg_l": bod_mg_l,

    "fecal_coliform_per_100ml":
        fecal_coliform_per_100ml,

    "total_coliform_per_100ml":
        total_coliform_per_100ml,

    "tds_mg_l": tds_mg_l,
    "nitrate_mg_l": nitrate_mg_l,
    "fluoride_mg_l": fluoride_mg_l,
    "arsenic_ug_l": arsenic_ug_l,

    "open_defecation_rate":
        open_defecation_rate,

    "toilet_access":
        toilet_access,

    "sewage_treatment_pct":
        sewage_treatment_pct,

    "handwashing_practice":
        handwashing_practice,

    "month": month,
    "season": season,

    "avg_temperature_c":
        avg_temperature_c,

    "avg_rainfall_mm":
        avg_rainfall_mm,

    "avg_humidity_pct":
        avg_humidity_pct,

    "flooding": flooding,

    "symptom_diarrhea":
        symptom_diarrhea,

    "symptom_vomiting":
        symptom_vomiting,

    "symptom_fever":
        symptom_fever,

    "symptom_abdominal_pain":
        symptom_abdominal_pain,

    "symptom_dehydration":
        symptom_dehydration,

    "symptom_jaundice":
        symptom_jaundice,

    "symptom_bloody_stool":
        symptom_bloody_stool,

    "symptom_skin_rash":
        symptom_skin_rash

}])


# ============================================================
# 7. FEATURE ENGINEERING
#    EXACTLY SAME AS TRAINING
# ============================================================

# Temporary coliform ratio

input_data["coliform_ratio"] = (
    input_data["fecal_coliform_per_100ml"]
    /
    (input_data["total_coliform_per_100ml"] + 1)
)


# Log transformed coliform ratio

input_data["log_coliform_ratio"] = np.log1p(
    input_data["coliform_ratio"]
)


# BOD / Dissolved Oxygen ratio

input_data["bod_do_ratio"] = (
    input_data["bod_mg_l"]
    /
    (input_data["dissolved_oxygen_mg_l"] + 1e-6)
)


# Sanitation risk

input_data["sanitation_risk"] = (
    input_data["open_defecation_rate"]
    *
    (
        1 -
        input_data["sewage_treatment_pct"] / 100
    )
)


# Remove temporary feature

input_data = input_data.drop(
    columns=["coliform_ratio"]
)


# ============================================================
# 8. ENSURE EXACT MODEL FEATURE ORDER
# ============================================================

input_data = input_data[
    model.feature_names_in_
]


# ============================================================
# 9. CHECK FEATURE COUNT
# ============================================================

print(
    "\nFeatures prepared for prediction:",
    input_data.shape[1]
)

if input_data.shape[1] != 43:

    print("\nERROR: Expected 43 features.")
    print(
        "Actual features:",
        input_data.shape[1]
    )

    exit()


# ============================================================
# 10. MAKE PREDICTION
# ============================================================

print("\n============================================================")
print("                    MAKING PREDICTION")
print("============================================================")

predicted_class = model.predict(input_data)[0]

probabilities = model.predict_proba(input_data)[0]

classes = model.classes_


# ============================================================
# 11. GET PREDICTED CLASS PROBABILITY
# ============================================================

predicted_index = np.where(
    classes == predicted_class
)[0][0]

predicted_probability = probabilities[
    predicted_index
]


# Get disease name

predicted_disease = disease_mapping[
    int(predicted_class)
]


# ============================================================
# 12. SORT TOP 3 PREDICTIONS
# ============================================================

sorted_indices = np.argsort(
    probabilities
)[::-1]


# ============================================================
# 13. DISPLAY FINAL RESULT
# ============================================================

print("\n============================================================")
print("                 PREDICTION RESULT")
print("============================================================")

print(
    "\nPredicted Disease :",
    predicted_disease
)

print(
    "Disease Class     :",
    predicted_class
)

print(
    "Model Probability :",
    f"{predicted_probability * 100:.2f}%"
)


# ============================================================
# 14. TOP 3 PREDICTIONS
# ============================================================

print("\n------------------------------------------------------------")
print("                    TOP 3 PREDICTIONS")
print("------------------------------------------------------------")

for rank, index in enumerate(
    sorted_indices[:3],
    start=1
):

    disease_class = int(classes[index])

    probability = probabilities[index]

    disease_name = disease_mapping[
        disease_class
    ]

    print(
        f"{rank}. {disease_name}"
        f" (Class {disease_class})"
        f" → {probability * 100:.2f}%"
    )


# ============================================================
# 15. ENGINEERED FEATURES
# ============================================================

print("\n------------------------------------------------------------")
print("                  ENGINEERED FEATURES")
print("------------------------------------------------------------")

print(
    "Log Coliform Ratio :",
    round(
        input_data[
            "log_coliform_ratio"
        ].iloc[0],
        4
    )
)

print(
    "BOD/DO Ratio       :",
    round(
        input_data[
            "bod_do_ratio"
        ].iloc[0],
        4
    )
)

print(
    "Sanitation Risk    :",
    round(
        input_data[
            "sanitation_risk"
        ].iloc[0],
        4
    )
)


# ============================================================
# 16. COMPLETION MESSAGE
# ============================================================

print("\n============================================================")
print("             PREDICTION COMPLETED SUCCESSFULLY")
print("============================================================")

print("\nDisease classification completed.")
print("The prediction uses environmental, water-quality,")
print("sanitation, demographic, climate and symptom features.")

print("\n============================================================")