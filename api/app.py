from flask import Flask, request, render_template
import pickle
import numpy as np
import os

# Initialize Flask app and set template folder
app = Flask(__name__, template_folder="templates")

# Load model and scaler safely
try:
    model_dir = os.path.join(os.path.dirname(__file__), "Model")
    scaler = pickle.load(open(os.path.join(model_dir, "standardScaler.pkl"), "rb"))
    model = pickle.load(open(os.path.join(model_dir, "modelForPrediction.pkl"), "rb"))
except Exception as e:
    print(f"Error loading model files: {e}")
    scaler = None
    model = None

# Home route → index.html
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Form route → predict.html
@app.route('/form', methods=['GET'])
def form():
    return render_template('predict.html')

# Prediction route → single_prediction.html
@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if not scaler or not model:
        return render_template('single_prediction.html',
                               error="Model files could not be loaded. Please check server logs.")

    try:
        # Extract and convert form data
        fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        data = [float(request.form.get(field)) for field in fields]

        # Scale and predict
        new_data = scaler.transform([data])
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
