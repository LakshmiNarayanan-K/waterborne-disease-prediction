import { useState } from "react";
import { STATES } from "./data/locations";
import "./App.css";

const months = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

const waterSources = [
  "Piped Water",
  "Tube Well",
  "Bore Well",
  "River",
  "Pond",
  "Lake",
  "Rainwater",
  "Other",
];

const waterTreatments = [
  "None",
  "Boiling",
  "Filtration",
  "Chlorination",
  "RO Purification",
  "Other",
];

function App() {
  const [selectedState, setSelectedState] = useState("");
  const [selectedDistrict, setSelectedDistrict] = useState("");

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const currentState = STATES.find(
    (state) => state.code === Number(selectedState)
  );

  const districts = currentState ? currentState.districts : [];

  const [form, setForm] = useState({
    state: "",
    district: "",
    age: "",
    gender: "",
    area: "",
    waterSource: "",
    waterTreatment: "",
    month: "",
    flooding: "",
    symptoms: {
      diarrhea: false,
      vomiting: false,
      fever: false,
      abdominalPain: false,
      dehydration: false,
      jaundice: false,
      bloodyStool: false,
      skinRash: false,
    },
  });

  const handleStateChange = (e) => {
    const value = e.target.value;

    setSelectedState(value);
    setSelectedDistrict("");

    setForm({
      ...form,
      state: value,
      district: "",
    });
  };

  const handleDistrictChange = (e) => {
    const value = e.target.value;

    setSelectedDistrict(value);

    setForm({
      ...form,
      district: value,
    });
  };

  const handleChange = (e) => {
    const { name, value } = e.target;

    setForm({
      ...form,
      [name]: value,
    });
  };

  const handleSymptomChange = (e) => {
    const { name, checked } = e.target;

    setForm({
      ...form,
      symptoms: {
        ...form.symptoms,
        [name]: checked,
      },
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    setLoading(true);
    setResult(null);
    setError("");

    try {
      // Convert human-readable form values
      // into the numeric values expected by the ML model.

      const payload = {
        state: Number(form.state),
        district: Number(form.district),

        age: Number(form.age),

        gender: form.gender === "Male" ? 0 : 1,

        is_urban: form.area === "Urban" ? 1 : 0,

        water_source: waterSources.indexOf(form.waterSource),

        water_treatment: waterTreatments.indexOf(
          form.waterTreatment
        ),

        // Current UI does not ask the user for WQI.
        // Use a neutral/default value for now.
        water_quality_index: 50,

        month: months.indexOf(form.month) + 1,

        flooding: form.flooding === "Yes" ? 1 : 0,

        symptom_diarrhea: form.symptoms.diarrhea ? 1 : 0,
        symptom_vomiting: form.symptoms.vomiting ? 1 : 0,
        symptom_fever: form.symptoms.fever ? 1 : 0,
        symptom_abdominal_pain:
          form.symptoms.abdominalPain ? 1 : 0,
        symptom_dehydration:
          form.symptoms.dehydration ? 1 : 0,
        symptom_jaundice:
          form.symptoms.jaundice ? 1 : 0,
        symptom_bloody_stool:
          form.symptoms.bloodyStool ? 1 : 0,
        symptom_skin_rash:
          form.symptoms.skinRash ? 1 : 0,
      };

      console.log("Sending to backend:", payload);

      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Prediction request failed"
        );
      }

      console.log("Backend response:", data);

      setResult(data);
    } catch (err) {
      console.error("Prediction error:", err);

      setError(
        err.message ||
          "Unable to connect to the prediction server."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="container">

        {/* HEADER */}
        <header className="header">
          <div className="badge">
            AI-POWERED HEALTH ANALYTICS
          </div>

          <h1>Waterborne Disease Prediction</h1>

          <p>
            Machine Learning based disease prediction for
            Northeast India
          </p>
        </header>

        <form onSubmit={handleSubmit}>

          {/* LOCATION */}
          <section className="card">
            <h2>📍 Location Information</h2>

            <div className="grid">

              <div className="field">
                <label>State</label>

                <select
                  value={selectedState}
                  onChange={handleStateChange}
                  required
                >
                  <option value="">
                    Select State
                  </option>

                  {STATES.map((state) => (
                    <option
                      key={state.code}
                      value={state.code}
                    >
                      {state.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="field">
                <label>District</label>

                <select
                  value={selectedDistrict}
                  onChange={handleDistrictChange}
                  disabled={!selectedState}
                  required
                >
                  <option value="">
                    {selectedState
                      ? "Select District"
                      : "Select State First"}
                  </option>

                  {districts.map((district) => (
                    <option
                      key={district.code}
                      value={district.code}
                    >
                      {district.name}
                    </option>
                  ))}
                </select>
              </div>

            </div>
          </section>

          {/* PATIENT */}
          <section className="card">
            <h2>👤 Patient Information</h2>

            <div className="grid">

              <div className="field">
                <label>Age</label>

                <input
                  type="number"
                  name="age"
                  value={form.age}
                  onChange={handleChange}
                  placeholder="Enter age"
                  min="1"
                  max="120"
                  required
                />
              </div>

              <div className="field">
                <label>Gender</label>

                <select
                  name="gender"
                  value={form.gender}
                  onChange={handleChange}
                  required
                >
                  <option value="">
                    Select Gender
                  </option>

                  <option value="Male">
                    Male
                  </option>

                  <option value="Female">
                    Female
                  </option>
                </select>
              </div>

              <div className="field">
                <label>Area Type</label>

                <select
                  name="area"
                  value={form.area}
                  onChange={handleChange}
                  required
                >
                  <option value="">
                    Select Area
                  </option>

                  <option value="Urban">
                    Urban
                  </option>

                  <option value="Rural">
                    Rural
                  </option>
                </select>
              </div>

            </div>
          </section>

          {/* WATER */}
          <section className="card">
            <h2>
              💧 Water & Environmental Information
            </h2>

            <div className="grid">

              <div className="field">
                <label>Water Source</label>

                <select
                  name="waterSource"
                  value={form.waterSource}
                  onChange={handleChange}
                  required
                >
                  <option value="">
                    Select Water Source
                  </option>

                  {waterSources.map((source) => (
                    <option
                      key={source}
                      value={source}
                    >
                      {source}
                    </option>
                  ))}
                </select>
              </div>

              <div className="field">
                <label>Water Treatment</label>

                <select
                  name="waterTreatment"
                  value={form.waterTreatment}
                  onChange={handleChange}
                  required
                >
                  <option value="">
                    Select Treatment
                  </option>

                  {waterTreatments.map(
                    (treatment) => (
                      <option
                        key={treatment}
                        value={treatment}
                      >
                        {treatment}
                      </option>
                    )
                  )}
                </select>
              </div>

              <div className="field">
                <label>Month</label>

                <select
                  name="month"
                  value={form.month}
                  onChange={handleChange}
                  required
                >
                  <option value="">
                    Select Month
                  </option>

                  {months.map((month) => (
                    <option
                      key={month}
                      value={month}
                    >
                      {month}
                    </option>
                  ))}
                </select>
              </div>

              <div className="field">
                <label>Flooding</label>

                <select
                  name="flooding"
                  value={form.flooding}
                  onChange={handleChange}
                  required
                >
                  <option value="">
                    Select Option
                  </option>

                  <option value="Yes">
                    Yes
                  </option>

                  <option value="No">
                    No
                  </option>
                </select>
              </div>

            </div>
          </section>

          {/* SYMPTOMS */}
          <section className="card">
            <h2>🩺 Symptoms</h2>

            <p className="hint">
              Select all symptoms currently observed.
            </p>

            <div className="symptoms">

              <label className="checkbox">
                <input
                  type="checkbox"
                  name="diarrhea"
                  checked={form.symptoms.diarrhea}
                  onChange={handleSymptomChange}
                />
                Diarrhea
              </label>

              <label className="checkbox">
                <input
                  type="checkbox"
                  name="vomiting"
                  checked={form.symptoms.vomiting}
                  onChange={handleSymptomChange}
                />
                Vomiting
              </label>

              <label className="checkbox">
                <input
                  type="checkbox"
                  name="fever"
                  checked={form.symptoms.fever}
                  onChange={handleSymptomChange}
                />
                Fever
              </label>

              <label className="checkbox">
                <input
                  type="checkbox"
                  name="abdominalPain"
                  checked={form.symptoms.abdominalPain}
                  onChange={handleSymptomChange}
                />
                Abdominal Pain
              </label>

              <label className="checkbox">
                <input
                  type="checkbox"
                  name="dehydration"
                  checked={form.symptoms.dehydration}
                  onChange={handleSymptomChange}
                />
                Dehydration
              </label>

              <label className="checkbox">
                <input
                  type="checkbox"
                  name="jaundice"
                  checked={form.symptoms.jaundice}
                  onChange={handleSymptomChange}
                />
                Jaundice
              </label>

              <label className="checkbox">
                <input
                  type="checkbox"
                  name="bloodyStool"
                  checked={form.symptoms.bloodyStool}
                  onChange={handleSymptomChange}
                />
                Bloody Stool
              </label>

              <label className="checkbox">
                <input
                  type="checkbox"
                  name="skinRash"
                  checked={form.symptoms.skinRash}
                  onChange={handleSymptomChange}
                />
                Skin Rash
              </label>

            </div>
          </section>

          {/* BUTTON */}
          <button
            className="predict-button"
            type="submit"
            disabled={loading}
          >
            {loading
              ? "🔄 Analyzing..."
              : "🔮 Predict Disease"}
          </button>

        </form>

        {/* ERROR */}
        {error && (
          <section className="card">
            <h2>⚠️ Prediction Error</h2>

            <p>{error}</p>
          </section>
        )}

        {/* RESULT */}
        {result && result.success && (
          <section className="card result-card">

            <h2>🧠 Prediction Result</h2>

            <div className="prediction-result">

              <h3>
                {result.prediction.disease}
              </h3>

              <p>
                Confidence:{" "}
                <strong>
                  {result.prediction.confidence}%
                </strong>
              </p>

            </div>

            <h3>Top Predictions</h3>

            <div className="top-predictions">

              {result.top_predictions.map(
                (prediction, index) => (
                  <div
                    className="prediction-item"
                    key={prediction.disease}
                  >
                    <span>
                      {index + 1}.{" "}
                      {prediction.disease}
                    </span>

                    <strong>
                      {prediction.confidence}%
                    </strong>
                  </div>
                )
              )}

            </div>

            {result.risk_indicators.length > 0 && (
              <>
                <h3>⚠️ Risk Indicators</h3>

                <ul>
                  {result.risk_indicators.map(
                    (risk) => (
                      <li key={risk}>
                        {risk}
                      </li>
                    )
                  )}
                </ul>
              </>
            )}

          </section>
        )}

        {/* FOOTER */}
        <footer>
          <p>
            AI-based prediction system • For educational
            and research purposes
          </p>
        </footer>

      </div>
    </div>
  );
}

export default App;