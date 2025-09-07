from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder='../')  # tell Flask to look one level up for HTML files

# Load model & scaler
model_path = os.path.join(os.path.dirname(__file__), "Model", "modelForPrediction.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "Model", "standardScaler.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')    

@app.route("/predictdata", methods=["POST"])
def predict():
    try:
        # Collect input values from form
        data = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"])
        ]
        # Scale & predict
        scaled = scaler.transform([data])
        prediction = model.predict(scaled)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


