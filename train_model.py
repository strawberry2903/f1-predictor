# train_model.py
"""
Train a RandomForest pipeline to predict 'podium' (top-3 finish).
Outputs:
 - model/f1_pipeline_compressed.pkl  # compressed for GitHub/Streamlit/HuggingFace
 - model/mappings.json
"""

import os, json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

DATA_DIR = "data"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1) Load CSVs
drivers = pd.read_csv(os.path.join(DATA_DIR, "drivers.csv"))
constructors = pd.read_csv(os.path.join(DATA_DIR, "constructors.csv"))
circuits = pd.read_csv(os.path.join(DATA_DIR, "circuits.csv"))
races = pd.read_csv(os.path.join(DATA_DIR, "races.csv"))
results = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))

# 2) Friendly driver names (for UI)
drivers["driver_name"] = (drivers["forename"].fillna("") + " " + drivers["surname"].fillna("")).str.strip()
driver_name_to_id = {str(name).strip().lower(): int(did) for did, name in zip(drivers["driverId"], drivers["driver_name"]) if pd.notna(name)}
constructor_name_to_id = {str(name).strip().lower(): int(cid) for cid, name in zip(constructors["constructorId"], constructors["name"]) if pd.notna(name)}
circuit_name_to_id = {str(name).strip().lower(): int(cid) for cid, name in zip(circuits["circuitId"], circuits["name"]) if pd.notna(name)}

drivers_list = sorted([n for n in drivers["driver_name"].dropna().unique()])
constructors_list = sorted([n for n in constructors["name"].dropna().unique()])
circuits_list = sorted([n for n in circuits["name"].dropna().unique()])

# 3) Merge results with race meta
races_small = races[["raceId", "year", "round", "circuitId", "date"]]
df = results.merge(races_small, on="raceId", how="left")
df["position"] = pd.to_numeric(df["position"], errors="coerce")
df = df.dropna(subset=["position"])
df["podium"] = (df["position"] <= 3).astype(int)

# 4) Features
df["grid"] = pd.to_numeric(df["grid"], errors="coerce").fillna(20).astype(int)
X = df[["driverId", "constructorId", "circuitId", "grid", "year"]].copy()
y = df["podium"]

# 5) Train-test split (last year as test)
last_year = int(df["year"].max())
train_mask = df["year"] < last_year
if train_mask.sum() < 100:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
else:
    X_train = X[train_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    X_test = X[~train_mask].reset_index(drop=True)
    y_test = y[~train_mask].reset_index(drop=True)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Last year (test): {last_year}")

# 6) Pipeline
categorical_features = ["driverId", "constructorId", "circuitId"]

preprocessor = ColumnTransformer(
    transformers=[
        ("enc", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features),
    ],
    remainder="passthrough",
)

pipeline = Pipeline(steps=[
    ("preproc", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=75, n_jobs=-1, random_state=42)),
])

# 7) Fit
pipeline.fit(X_train, y_train)

# 8) Evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", acc)
print("F1 score:", f1)
print(classification_report(y_test, y_pred, digits=3))

# 9) Save compressed model + mappings
joblib.dump(pipeline, os.path.join(MODEL_DIR, "f1_pipeline_compressed.pkl"), compress=9)
mappings = {
    "driver_name_to_id": driver_name_to_id,
    "constructor_name_to_id": constructor_name_to_id,
    "circuit_name_to_id": circuit_name_to_id,
    "drivers_list": drivers_list,
    "constructors_list": constructors_list,
    "circuits_list": circuits_list,
    "default_year": last_year,
    "metrics": {"accuracy": float(acc), "f1": float(f1), "test_year": last_year}
}
with open(os.path.join(MODEL_DIR, "mappings.json"), "w") as f:
    json.dump(mappings, f)

print("Saved pipeline and mappings to 'model/'")
