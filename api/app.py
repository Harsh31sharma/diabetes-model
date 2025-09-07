import os
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__, template_folder="../templates")

# Paths to your serialized artifacts
BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, "Model")
model_path = os.path.join(MODEL_DIR, "modelForPrediction.pkl")
scaler_path= os.path.join(MODEL_DIR, "standardScaler.pkl")

# Load model and scaler with error logging
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    app.logger.error(f"Failed to load model/scaler: {e}")
    raise

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predictdata", methods=["POST"])
def predict_datapoint():
    try:
        # Collect and scale inputs
        keys   = [
            "Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"
        ]
        values = [float(request.form.get(k, 0)) for k in keys]
        data   = scaler.transform([values])

        # Predict and calculate confidence
        pred = model.predict(data)[0]
        conf = model.predict_proba(data)[0][1] * 100

        result = "Diabetic" if pred == 1 else "Non-Diabetic"
        return render_template(
            "single_prediction.html",
            result=result,
            confidence=round(conf, 2)
        )
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return render_template(
            "single_prediction.html",
            error="Invalid input or server error."
        ), 500
