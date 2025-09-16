# app.py
import streamlit as st
import pandas as pd
import joblib
import json
import os

# Page setup
st.set_page_config(page_title="F1 Podium Predictor", layout="centered")
st.title("🏁 F1 Podium Predictor (podium vs not)")

# Paths
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "f1_pipeline_compressed.pkl")  # updated to compressed model
MAP_PATH = os.path.join(MODEL_DIR, "mappings.json")

# Check if files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(MAP_PATH):
    st.error("Model or mappings not found. Run `train_model.py` first and ensure model/ exists.")
    st.stop()

# Load model and mappings
model = joblib.load(MODEL_PATH)
with open(MAP_PATH, "r") as f:
    mappings = json.load(f)

# UI Controls
driver = st.selectbox("Driver", mappings["drivers_list"])
constructor = st.selectbox("Constructor", mappings["constructors_list"])
circuit = st.selectbox("Circuit", mappings["circuits_list"])
grid = st.number_input("Grid position (starting slot)", min_value=1, max_value=30, value=10, step=1)
year = st.number_input(
    "Year (use latest or override)",
    min_value=1950,
    max_value=2100,
    value=mappings.get("default_year", 2023),
    step=1
)

st.markdown("---")
st.write("Model test performance (on held-out year):")
st.write(
    f"Accuracy: **{mappings['metrics']['accuracy']:.3f}**, "
    f"F1: **{mappings['metrics']['f1']:.3f}**, "
    f"test year: **{mappings['metrics']['test_year']}**"
)

# Prediction button
if st.button("Predict podium probability"):
    # map human names -> ids
    driver_id = mappings["driver_name_to_id"].get(driver.strip().lower())
    ctor_id = mappings["constructor_name_to_id"].get(constructor.strip().lower())
    circuit_id = mappings["circuit_name_to_id"].get(circuit.strip().lower())

    if driver_id is None or ctor_id is None or circuit_id is None:
        st.error("Could not map driver/constructor/circuit to ids. Check your mappings.")
    else:
        row = pd.DataFrame([{
            "driverId": int(driver_id),
            "constructorId": int(ctor_id),
            "circuitId": int(circuit_id),
            "grid": int(grid),
            "year": int(year)
        }])
        proba = model.predict_proba(row)[0][1]  # probability of class '1' (podium)
        pct = proba * 100
        st.success(f"Estimated podium probability: **{pct:.1f}%**")
        st.progress(proba)  # visual indicator

        # Two-bar chart
        st.bar_chart(pd.DataFrame({"chance": [1 - proba, proba]}, index=["Not Podium", "Podium"]))
        st.write("Interpretation: model predicts probability that this driver finishes top-3 given the inputs.")
