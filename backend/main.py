from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

from feature_engineering import add_features


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Waterborne Disease Prediction API",
    description="RF + XGBoost ensemble prediction API",
    version="1.0"
)


# ============================================================
# CORS - ALLOW REACT FRONTEND
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# FILES
# ============================================================

DATASET = "X_train.csv"
RF_MODEL = "waterborne_disease_rf_model.pkl"
XGB_MODEL = "waterborne_disease_xgboost_model.pkl"


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

print("Loading dataset...")

lookup_df = pd.read_csv(DATASET)

print("Loading Random Forest model...")
rf_model = joblib.load(RF_MODEL)

print("Loading XGBoost model...")
xgb_model = joblib.load(XGB_MODEL)

print("Models loaded successfully.")


# ============================================================
# DISEASE NAMES
# ============================================================

DISEASE_NAMES = {
    0: "Cholera",
    1: "Dysentery",
    2: "Giardiasis",
    3: "Hepatitis A",
    4: "Hepatitis E",
    5: "Leptospirosis",
    6: "No Disease",
    7: "Typhoid"
}


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionInput(BaseModel):
    state: int
    district: int

    age: int
    gender: int
    is_urban: int

    water_source: int
    water_treatment: int
    water_quality_index: float

    month: int
    flooding: int

    symptom_diarrhea: int
    symptom_vomiting: int
    symptom_fever: int
    symptom_abdominal_pain: int
    symptom_dehydration: int
    symptom_jaundice: int
    symptom_bloody_stool: int
    symptom_skin_rash: int


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_median_value(column, state, district, month):
    """
    Find the most relevant median value using:

    1. District + Month
    2. District
    3. State + Month
    4. State
    5. Overall dataset
    """

    # District + Month
    result = lookup_df[
        (lookup_df["state"] == state) &
        (lookup_df["district"] == district) &
        (lookup_df["month"] == month)
    ][column]

    if not result.empty:
        return float(result.median())

    # District
    result = lookup_df[
        (lookup_df["state"] == state) &
        (lookup_df["district"] == district)
    ][column]

    if not result.empty:
        return float(result.median())

    # State + Month
    result = lookup_df[
        (lookup_df["state"] == state) &
        (lookup_df["month"] == month)
    ][column]

    if not result.empty:
        return float(result.median())

    # State
    result = lookup_df[
        lookup_df["state"] == state
    ][column]

    if not result.empty:
        return float(result.median())

    # Overall
    return float(lookup_df[column].median())


def get_mode_value(column, state, district):
    """
    Find the most common categorical value.
    """

    # District
    result = lookup_df[
        (lookup_df["state"] == state) &
        (lookup_df["district"] == district)
    ][column]

    if not result.empty:
        return result.mode().iloc[0]

    # State
    result = lookup_df[
        lookup_df["state"] == state
    ][column]

    if not result.empty:
        return result.mode().iloc[0]

    # Overall
    return lookup_df[column].mode().iloc[0]


def get_season(month):
    """
    Dataset season encoding.
    """

    if month in [12, 1, 2]:
        return 0

    elif month in [3, 4, 5]:
        return 1

    elif month in [6, 7, 8, 9]:
        return 2

    else:
        return 3


# ============================================================
# HEALTH RISK INDICATORS
# ============================================================

def get_risk_indicators(row):

    indicators = []

    if row["fecal_coliform_per_100ml"] > 500:
        indicators.append("High fecal coliform level")

    if row["turbidity_ntu"] > 5:
        indicators.append("High water turbidity")

    if row["open_defecation_rate"] > 20:
        indicators.append("High open defecation rate")

    if row["sewage_treatment_pct"] < 50:
        indicators.append("Low sewage treatment coverage")

    if row["flooding"] == 1:
        indicators.append("Flooding reported")

    return indicators


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Waterborne Disease Prediction API is running",
        "status": "online"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(data: PredictionInput):

    state = data.state
    district = data.district
    month = data.month

    # --------------------------------------------------------
    # Automatically obtain environmental values
    # --------------------------------------------------------

    latitude = get_median_value(
        "latitude",
        state,
        district,
        month
    )

    longitude = get_median_value(
        "longitude",
        state,
        district,
        month
    )

    population_density = get_median_value(
        "population_density",
        state,
        district,
        month
    )

    # --------------------------------------------------------
    # WATER QUALITY
    # --------------------------------------------------------

    ph = get_median_value(
        "ph",
        state,
        district,
        month
    )

    turbidity = get_median_value(
        "turbidity_ntu",
        state,
        district,
        month
    )

    dissolved_oxygen = get_median_value(
        "dissolved_oxygen_mg_l",
        state,
        district,
        month
    )

    bod = get_median_value(
        "bod_mg_l",
        state,
        district,
        month
    )

    fecal_coliform = get_median_value(
        "fecal_coliform_per_100ml",
        state,
        district,
        month
    )

    total_coliform = get_median_value(
        "total_coliform_per_100ml",
        state,
        district,
        month
    )

    tds = get_median_value(
        "tds_mg_l",
        state,
        district,
        month
    )

    nitrate = get_median_value(
        "nitrate_mg_l",
        state,
        district,
        month
    )

    fluoride = get_median_value(
        "fluoride_mg_l",
        state,
        district,
        month
    )

    arsenic = get_median_value(
        "arsenic_ug_l",
        state,
        district,
        month
    )

    # --------------------------------------------------------
    # SANITATION
    # --------------------------------------------------------

    open_defecation = get_median_value(
        "open_defecation_rate",
        state,
        district,
        month
    )

    toilet_access = get_median_value(
        "toilet_access",
        state,
        district,
        month
    )

    sewage_treatment = get_median_value(
        "sewage_treatment_pct",
        state,
        district,
        month
    )

    handwashing = get_mode_value(
        "handwashing_practice",
        state,
        district
    )

    # --------------------------------------------------------
    # CLIMATE
    # --------------------------------------------------------

    temperature = get_median_value(
        "avg_temperature_c",
        state,
        district,
        month
    )

    rainfall = get_median_value(
        "avg_rainfall_mm",
        state,
        district,
        month
    )

    humidity = get_median_value(
        "avg_humidity_pct",
        state,
        district,
        month
    )

    # --------------------------------------------------------
    # REGION AND SEASON
    # --------------------------------------------------------

    region = 0

    season = get_season(month)

    # --------------------------------------------------------
    # CONSTRUCT ORIGINAL 40 FEATURES
    # --------------------------------------------------------

    input_data = {

        "state": state,
        "district": district,
        "region": region,

        "latitude": latitude,
        "longitude": longitude,

        "is_urban": data.is_urban,
        "population_density": population_density,

        "age": data.age,
        "gender": data.gender,

        "water_source": data.water_source,
        "water_treatment": data.water_treatment,
        "water_quality_index": data.water_quality_index,

        "ph": ph,
        "turbidity_ntu": turbidity,
        "dissolved_oxygen_mg_l": dissolved_oxygen,
        "bod_mg_l": bod,

        "fecal_coliform_per_100ml": fecal_coliform,
        "total_coliform_per_100ml": total_coliform,

        "tds_mg_l": tds,
        "nitrate_mg_l": nitrate,
        "fluoride_mg_l": fluoride,
        "arsenic_ug_l": arsenic,

        "open_defecation_rate": open_defecation,
        "toilet_access": toilet_access,
        "sewage_treatment_pct": sewage_treatment,
        "handwashing_practice": handwashing,

        "month": month,
        "season": season,

        "avg_temperature_c": temperature,
        "avg_rainfall_mm": rainfall,
        "avg_humidity_pct": humidity,

        "flooding": data.flooding,

        "symptom_diarrhea": data.symptom_diarrhea,
        "symptom_vomiting": data.symptom_vomiting,
        "symptom_fever": data.symptom_fever,
        "symptom_abdominal_pain": data.symptom_abdominal_pain,
        "symptom_dehydration": data.symptom_dehydration,
        "symptom_jaundice": data.symptom_jaundice,
        "symptom_bloody_stool": data.symptom_bloody_stool,
        "symptom_skin_rash": data.symptom_skin_rash
    }

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame([input_data])

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    df = add_features(df)

    # --------------------------------------------------------
    # EXACT MODEL FEATURE ORDER
    # --------------------------------------------------------

    feature_names = rf_model.feature_names_in_

    df = df[feature_names]

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf_probability = rf_model.predict_proba(df)[0]

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    xgb_probability = xgb_model.predict_proba(df)[0]

    # --------------------------------------------------------
    # ENSEMBLE
    # --------------------------------------------------------

    ensemble_probability = (
        0.5 * rf_probability +
        0.5 * xgb_probability
    )

    predicted_class = int(
        np.argmax(ensemble_probability)
    )

    confidence = float(
        ensemble_probability[predicted_class] * 100
    )

    # --------------------------------------------------------
    # TOP 3 PREDICTIONS
    # --------------------------------------------------------

    top_indices = np.argsort(
        ensemble_probability
    )[::-1][:3]

    top_predictions = []

    for index in top_indices:

        top_predictions.append({
            "disease": DISEASE_NAMES[int(index)],
            "confidence": round(
                float(
                    ensemble_probability[index] * 100
                ),
                2
            )
        })

    # --------------------------------------------------------
    # RISK INDICATORS
    # --------------------------------------------------------

    risk_indicators = get_risk_indicators(
        input_data
    )

    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    return {

        "success": True,

        "prediction": {
            "class": predicted_class,
            "disease": DISEASE_NAMES[predicted_class],
            "confidence": round(
                confidence,
                2
            )
        },

        "top_predictions": top_predictions,

        "risk_indicators": risk_indicators,

        "engineered_features": {

            "log_coliform_ratio": round(
                float(
                    df["log_coliform_ratio"].iloc[0]
                ),
                4
            ),

            "bod_do_ratio": round(
                float(
                    df["bod_do_ratio"].iloc[0]
                ),
                4
            ),

            "sanitation_risk": round(
                float(
                    df["sanitation_risk"].iloc[0]
                ),
                4
            )
        }
    }