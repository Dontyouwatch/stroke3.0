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

        # Prepare input features with exact expected order
        bmi = float(data.get("BMI", 0))
        age = float(data.get("Age", 0))
        cholesterol = float(data.get("Cholesterol", 0))
        hypertension = int(data.get("Hypertension_Category", 0))
        atrial_fib = int(data.get("Atrial_Fibrillation", 0))

        # Create input data with exact order from your model
        input_data = {
            "Atrial_Fibrillation": atrial_fib,
            "Age": age,
            "Diabetes": int(data.get("Diabetes", 0)),
            "Cholesterol": cholesterol,
            "Smoking": int(data.get("Smoking", 0)),
            "BMI": bmi,
            "Previous_Stroke": int(data.get("Previous_Stroke", 0)),
            "Hypertension_Category": hypertension,
            "BMI_Hypertension": float(bmi * hypertension),  # Explicit float conversion
            "Cholesterol_Atrial_Fibrillation": float(cholesterol * atrial_fib),
            "BMI_Category_1": 1 if 18.5 <= bmi < 24.9 else 0,
            "BMI_Category_2": 1 if 25 <= bmi < 29.9 else 0,
            "BMI_Category_3": 1 if bmi >= 30 else 0,
            "Cholesterol_Category_1": 1 if 200 <= cholesterol < 240 else 0,
            "Cholesterol_Category_2": 1 if cholesterol >= 240 else 0,
            "Age_Group_1": 1 if 40 <= age < 60 else 0,
            "Age_Group_2": 1 if age >= 60 else 0
        }

        # Create DataFrame ensuring exact column order
        feature_order = [
            'Atrial_Fibrillation', 'Age', 'Diabetes', 'Cholesterol', 
            'Smoking', 'BMI', 'Previous_Stroke', 'Hypertension_Category',
            'BMI_Hypertension', 'Cholesterol_Atrial_Fibrillation',
            'BMI_Category_1', 'BMI_Category_2', 'BMI_Category_3',
            'Cholesterol_Category_1', 'Cholesterol_Category_2',
            'Age_Group_1', 'Age_Group_2'
        ]
        input_df = pd.DataFrame([input_data])[feature_order]

        # Scale numeric features
        numeric_features = ['Age', 'Cholesterol', 'BMI', 'BMI_Hypertension', 'Cholesterol_Atrial_Fibrillation']
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])

        # Make prediction and ensure JSON-serializable output
        stroke_prob = float(model.predict_proba(input_df)[0][1])  # Convert numpy float32 to Python float
        
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
