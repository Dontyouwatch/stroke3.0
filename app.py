from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path

app = Flask(__name__)

# Initialize model and scaler
model = None
scaler = None

def load_model_and_scaler():
    """Load the trained model and scaler from files"""
    global model, scaler
    
    try:
        # Load the trained model
        model_path = Path(__file__).parent / 'models' / 'strokemodel.pkl'
        model = joblib.load(model_path)
        
        # Load the scaler
        scaler_path = Path(__file__).parent / 'models' / 'scaler.pkl'
        scaler = joblib.load(scaler_path)
        
        print("Model and scaler loaded successfully")
    except Exception as e:
        print(f"Error loading model or scaler: {str(e)}")
        raise

# Load model and scaler when the app starts
load_model_and_scaler()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['Age', 'BMI', 'Cholesterol', 'Hypertension_Category', 
                         'Atrial_Fibrillation', 'Diabetes', 'Smoking', 'Previous_Stroke']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}', 'status': 'error'}), 400
        
        # Calculate BMI categories (fixed syntax)
        bmi = float(data["BMI"])
        bmi_cat_1 = 1 if 18.5 <= bmi < 24.9 else 0
        bmi_cat_2 = 1 if 25 <= bmi < 29.9 else 0
        bmi_cat_3 = 1 if bmi >= 30 else 0
        
        # Calculate cholesterol categories
        cholesterol = float(data["Cholesterol"])
        chol_cat_1 = 1 if 200 <= cholesterol < 240 else 0
        chol_cat_2 = 1 if cholesterol >= 240 else 0
        
        # Calculate age groups
        age = float(data["Age"])
        age_group_1 = 1 if 40 <= age < 60 else 0
        age_group_2 = 1 if age >= 60 else 0
        
        # Prepare input data as DataFrame with all expected features
        input_data = pd.DataFrame([{
            "Age": age,
            "BMI": bmi,
            "Cholesterol": cholesterol,
            "Hypertension_Category": int(data["Hypertension_Category"]),
            "Atrial_Fibrillation": int(data["Atrial_Fibrillation"]),
            "Diabetes": int(data["Diabetes"]),
            "Smoking": int(data["Smoking"]),
            "Previous_Stroke": int(data["Previous_Stroke"]),
            "BMI_Category_1": bmi_cat_1,
            "BMI_Category_2": bmi_cat_2,
            "BMI_Category_3": bmi_cat_3,
            "Cholesterol_Category_1": chol_cat_1,
            "Cholesterol_Category_2": chol_cat_2,
            "Age_Group_1": age_group_1,
            "Age_Group_2": age_group_2,
            "BMI_Hypertension": bmi * int(data["Hypertension_Category"]),
            "Cholesterol_Atrial_Fibrillation": cholesterol * int(data["Atrial_Fibrillation"])
        }])
        
        # Define numeric features for scaling
        numeric_features = ["Age", "Cholesterol", "BMI", "BMI_Hypertension", "Cholesterol_Atrial_Fibrillation"]
        
        # Scale the numeric features
        input_data[numeric_features] = scaler.transform(input_data[numeric_features])
        
        # Expected columns in correct order
        expected_columns = [
            'Age', 'BMI', 'Cholesterol', 'Hypertension_Category', 'Atrial_Fibrillation',
            'Diabetes', 'Smoking', 'Previous_Stroke', 'BMI_Category_1', 'BMI_Category_2',
            'BMI_Category_3', 'Cholesterol_Category_1', 'Cholesterol_Category_2',
            'Age_Group_1', 'Age_Group_2', 'BMI_Hypertension', 'Cholesterol_Atrial_Fibrillation'
        ]
        
        # Ensure correct column order
        input_data = input_data[expected_columns]
        
        # Make prediction
        stroke_prob = model.predict_proba(input_data)[0][1]
        
        # Prepare response
        response = {
            'name': data.get('name', 'User'),
            'risk_percentage': round(stroke_prob * 100, 1),
            'status': 'success'
        }
        
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'message': 'An error occurred during prediction'
        }), 500

@app.route('/')
def home():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
