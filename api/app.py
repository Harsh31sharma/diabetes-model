# api/predict.py
from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder="templates")

# Wrap model loading in try/except
try:
    model_dir = os.path.join(os.path.dirname(__file__), "Model")
    scaler = pickle.load(open(os.path.join(model_dir, "standardScaler.pkl"), "rb"))
    model = pickle.load(open(os.path.join(model_dir, "modelForPrediction.pkl"), "rb"))

except Exception as e:
    print(f"Error loading model files: {e}")
    scaler = None
    model = None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if not scaler or not model:
        return render_template('single_prediction.html',
                               error="Model files could not be loaded. Please check server logs.")

    try:
        Pregnancies = float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                      Insulin, BMI, DiabetesPedigreeFunction, Age]])
        prediction = model.predict(new_data)[0]
        confidence = model.predict_proba(new_data)[0][1] * 100

        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return render_template('single_prediction.html',
                               result=result,
                               confidence=round(confidence, 2))
    except ValueError as ve:
        return render_template('single_prediction.html',
                               error=f"Invalid input: {ve}")
    except Exception as e:
        return render_template('single_prediction.html',
                               error=f"Prediction failed: {e}")

