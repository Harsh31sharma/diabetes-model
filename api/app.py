# api/predict.py
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder="../templates")

model_path = os.path.join(os.path.dirname(__file__), "Model", "modelForPrediction.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "Model", "standardScaler.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
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
    except:
        return render_template('single_prediction.html',
                               error="Invalid input. Please enter numeric values only.")
