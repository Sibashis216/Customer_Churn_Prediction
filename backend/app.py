"""
Customer Churn Prediction — backend API.

Loads the trained Random Forest model + label encoders and exposes a single
prediction endpoint consumed by the static frontend in ../frontend.
"""
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
CORS(app)  # allow the frontend (served separately / from file://) to call this API

# ---------------------------------------------------------------------------
# Load model + encoders once at startup
# ---------------------------------------------------------------------------
with open(BASE_DIR / "customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

MODEL = model_data["model"]
FEATURE_NAMES = model_data["features_names"]

with open(BASE_DIR / "encoders.pkl", "rb") as f:
    _raw_encoders = pickle.load(f)

# Only keep encoders for columns that are actually model input features
# (defensive against a stray "Churn" target-encoder ending up in the pickle).
ENCODERS = {col: enc for col, enc in _raw_encoders.items() if col in FEATURE_NAMES}
CATEGORICAL_FEATURES = list(ENCODERS.keys())
NUMERIC_FEATURES = [f for f in FEATURE_NAMES if f not in CATEGORICAL_FEATURES]

# Values the frontend is allowed to send for each categorical field, taken
# straight from the encoders that were fit during training.
FIELD_OPTIONS = {col: list(le.classes_) for col, le in ENCODERS.items()}


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model": "RandomForestClassifier"})


@app.get("/api/schema")
def schema():
    """Lets the frontend build its form dynamically from the real training columns."""
    return jsonify(
        {
            "feature_order": FEATURE_NAMES,
            "categorical_options": FIELD_OPTIONS,
            "numeric_features": NUMERIC_FEATURES,
        }
    )


def _validate_and_prepare(payload: dict) -> pd.DataFrame:
    missing = [f for f in FEATURE_NAMES if f not in payload]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    row = {}
    for feature in FEATURE_NAMES:
        value = payload[feature]

        if feature == "SeniorCitizen":
            row[feature] = int(value)
        elif feature in NUMERIC_FEATURES:
            row[feature] = float(value)
        else:  # categorical -> must match one of the trained classes
            valid_values = FIELD_OPTIONS[feature]
            if value not in valid_values:
                raise ValueError(
                    f"Invalid value '{value}' for '{feature}'. Expected one of: {valid_values}"
                )
            row[feature] = value

    df = pd.DataFrame([row])[FEATURE_NAMES]

    for column, encoder in ENCODERS.items():
        df[column] = encoder.transform(df[column])

    return df


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be JSON."}), 400

    try:
        df = _validate_and_prepare(payload)
    except (ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400

    probability_churn = float(MODEL.predict_proba(df)[0][1])
    prediction = int(MODEL.predict(df)[0])

    # Give a lightweight explanation using the model's global feature importances,
    # scoped to the fields this customer actually has "risk" values for.
    importances = dict(zip(FEATURE_NAMES, MODEL.feature_importances_))
    top_features = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:5]

    return jsonify(
        {
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "will_churn": bool(prediction),
            "churn_probability": round(probability_churn, 4),
            "retain_probability": round(1 - probability_churn, 4),
            "top_drivers": [{"feature": f, "importance": round(v, 4)} for f, v in top_features],
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
