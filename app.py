from flask import Flask, request, jsonify, send_from_directory
from xgboost import XGBClassifier
import pandas as pd
import joblib
from pathlib import Path
import os
import traceback

app = Flask(__name__, static_folder='static')

# Initialize model and scaler
model = None
scaler = None

def load_model_and_scaler():
    """Load the trained model (JSON) and scaler (pkl)"""
    global model, scaler
    
    try:
        model_dir = Path(__file__).parent / 'models'
        model_path = model_dir / 'strokemodel.json'
        scaler_path = model_dir / 'scaler.pkl'
        
        # Load XGBoost model from JSON
        model = XGBClassifier()
        model.load_model(model_path)
        
        # Load scaler from pickle
        scaler = joblib.load(scaler_path)
        
        print("Model and scaler loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model or scaler: {str(e)}")
        print(traceback.format_exc())
        return False

# Load model at startup
if not load_model_and_scaler():
    print("Failed to load model - check error messages above")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error',
                'message': 'Prediction service unavailable'
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No data received',
                'status': 'error'
            }), 400

        # Prepare input features
        bmi = float(data.get("BMI", 0))
        age = float(data.get("Age", 0))
        cholesterol = float(data.get("Cholesterol", 0))
        
        input_features = {
            "Age": age,
            "BMI": bmi,
            "Cholesterol": cholesterol,
            "Hypertension_Category": int(data.get("Hypertension_Category", 0)),
            "Atrial_Fibrillation": int(data.get("Atrial_Fibrillation", 0)),
            "Diabetes": int(data.get("Diabetes", 0)),
            "Smoking": int(data.get("Smoking", 0)),
            "Previous_Stroke": int(data.get("Previous_Stroke", 0)),
            "BMI_Category_1": 1 if 18.5 <= bmi < 24.9 else 0,
            "BMI_Category_2": 1 if 25 <= bmi < 29.9 else 0,
            "BMI_Category_3": 1 if bmi >= 30 else 0,
            "Cholesterol_Category_1": 1 if 200 <= cholesterol < 240 else 0,
            "Cholesterol_Category_2": 1 if cholesterol >= 240 else 0,
            "Age_Group_1": 1 if 40 <= age < 60 else 0,
            "Age_Group_2": 1 if age >= 60 else 0,
            "BMI_Hypertension": bmi * int(data.get("Hypertension_Category", 0)),
            "Cholesterol_Atrial_Fibrillation": cholesterol * int(data.get("Atrial_Fibrillation", 0))
        }

        # Create DataFrame
        input_df = pd.DataFrame([input_features])
        
        # Scale numeric features
        numeric_features = ["Age", "Cholesterol", "BMI", "BMI_Hypertension", "Cholesterol_Atrial_Fibrillation"]
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])
        
        # Ensure correct column order
        expected_columns = [
            'Age', 'BMI', 'Cholesterol', 'Hypertension_Category', 'Atrial_Fibrillation',
            'Diabetes', 'Smoking', 'Previous_Stroke', 'BMI_Category_1', 'BMI_Category_2',
            'BMI_Category_3', 'Cholesterol_Category_1', 'Cholesterol_Category_2',
            'Age_Group_1', 'Age_Group_2', 'BMI_Hypertension', 'Cholesterol_Atrial_Fibrillation'
        ]
        input_df = input_df[expected_columns]
        
        # Make prediction
        stroke_prob = model.predict_proba(input_df)[0][1]
        
        return jsonify({
            'risk_percentage': round(stroke_prob * 100, 1),
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
