import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
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
    """Load the trained model and scaler from files"""
    global model, scaler
    
    try:
        model_dir = Path(__file__).parent / 'models'
        
        # Load model - try both pickle and native XGBoost format
        model_path = model_dir / 'strokemodel.pkl'
        if not model_path.exists():
            model_path = model_dir / 'strokemodel.json'
        
        if model_path.suffix == '.pkl':
            model = joblib.load(model_path)
        else:
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(model_path)
        
        # Load scaler
        scaler_path = model_dir / 'scaler.pkl'
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
        
        # Validate all required fields with type checking
        required_fields = {
            'Age': (float, 18, 120),
            'BMI': (float, 10, 50),
            'Cholesterol': (float, 100, 400),
            'Hypertension_Category': (int, 0, 2),
            'Atrial_Fibrillation': (int, 0, 1),
            'Diabetes': (int, 0, 1),
            'Smoking': (int, 0, 1),
            'Previous_Stroke': (int, 0, 1)
        }
        
        errors = []
        processed_data = {}
        
        for field, (dtype, min_val, max_val) in required_fields.items():
            if field not in data:
                errors.append(f"Missing required field: {field}")
                continue
            
            try:
                value = dtype(data[field])
                if not (min_val <= value <= max_val):
                    errors.append(f"{field} must be between {min_val} and {max_val}")
                processed_data[field] = value
            except (ValueError, TypeError):
                errors.append(f"Invalid value for {field}")
        
        if errors:
            return jsonify({
                'errors': errors,
                'status': 'error'
            }), 400
        
        # Feature engineering
        bmi = processed_data['BMI']
        age = processed_data['Age']
        cholesterol = processed_data['Cholesterol']
        
        features = {
            "Age": age,
            "BMI": bmi,
            "Cholesterol": cholesterol,
            "Hypertension_Category": processed_data['Hypertension_Category'],
            "Atrial_Fibrillation": processed_data['Atrial_Fibrillation'],
            "Diabetes": processed_data['Diabetes'],
            "Smoking": processed_data['Smoking'],
            "Previous_Stroke": processed_data['Previous_Stroke'],
            "BMI_Category_1": 1 if 18.5 <= bmi < 24.9 else 0,
            "BMI_Category_2": 1 if 25 <= bmi < 29.9 else 0,
            "BMI_Category_3": 1 if bmi >= 30 else 0,
            "Cholesterol_Category_1": 1 if 200 <= cholesterol < 240 else 0,
            "Cholesterol_Category_2": 1 if cholesterol >= 240 else 0,
            "Age_Group_1": 1 if 40 <= age < 60 else 0,
            "Age_Group_2": 1 if age >= 60 else 0,
            "BMI_Hypertension": bmi * processed_data['Hypertension_Category'],
            "Cholesterol_Atrial_Fibrillation": cholesterol * processed_data['Atrial_Fibrillation']
        }
        
        # Create DataFrame and scale features
        input_df = pd.DataFrame([features])
        numeric_cols = ["Age", "Cholesterol", "BMI", "BMI_Hypertension", "Cholesterol_Atrial_Fibrillation"]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        
        # Prediction
        stroke_prob = model.predict_proba(input_df)[0][1]
        
        return jsonify({
            'risk_percentage': round(stroke_prob * 100, 1),
            'status': 'success'
        })
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error',
            'message': 'Prediction failed'
        }), 500

# Static file routes
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
