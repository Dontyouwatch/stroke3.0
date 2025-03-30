from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Initialize model and scaler
model = None
scaler = None

def load_model_and_scaler():
    """Load the trained model and scaler from files"""
    global model, scaler
    
    try:
        # Load the trained model
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'strokemodel.pkl')
        model = joblib.load(model_path)
        
        # Load the scaler
        scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
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
        
        # Prepare input data as DataFrame with all expected features
        input_data = pd.DataFrame([{
            "Age": float(data["Age"]),
            "BMI": float(data["BMI"]),
            "Cholesterol": float(data["Cholesterol"]),
            "Hypertension_Category": int(data["Hypertension_Category"]),
            "Atrial_Fibrillation": int(data["Atrial_Fibrillation"]),
            "Diabetes": int(data["Diabetes"]),
            "Smoking": int(data["Smoking"]),
            "Previous_Stroke": int(data["Previous_Stroke"]),
            "BMI_Category_1": 1 if (18.5 <= float(data["BMI"]) < 24.9 else 0,
            "BMI_Category_2": 1 if (25 <= float(data["BMI"]) < 29.9 else 0,
            "BMI_Category_3": 1 if float(data["BMI"]) >= 30 else 0,
            "Cholesterol_Category_1": 1 if (200 <= float(data["Cholesterol"]) < 240 else 0,
            "Cholesterol_Category_2": 1 if float(data["Cholesterol"]) >= 240 else 0,
            "Age_Group_1": 1 if (40 <= float(data["Age"]) < 60 else 0,
            "Age_Group_2": 1 if float(data["Age"]) >= 60 else 0,
            "BMI_Hypertension": float(data["BMI"]) * int(data["Hypertension_Category"]),
            "Cholesterol_Atrial_Fibrillation": float(data["Cholesterol"]) * int(data["Atrial_Fibrillation"])
        }])
        
        # Define numeric features for scaling (must match your training)
        numeric_features = ["Age", "Cholesterol", "BMI", "BMI_Hypertension", "Cholesterol_Atrial_Fibrillation"]
        
        # Scale the numeric features
        input_data[numeric_features] = scaler.transform(input_data[numeric_features])
        
        # Ensure all expected columns are present
        expected_columns = ['Age', 'BMI', 'Cholesterol', 'Hypertension_Category', 'Atrial_Fibrillation',
                           'Diabetes', 'Smoking', 'Previous_Stroke', 'BMI_Category_1', 'BMI_Category_2',
                           'BMI_Category_3', 'Cholesterol_Category_1', 'Cholesterol_Category_2',
                           'Age_Group_1', 'Age_Group_2', 'BMI_Hypertension', 'Cholesterol_Atrial_Fibrillation']
        
        # Add any missing columns with 0 values
        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match training data
        input_data = input_data[expected_columns]
        
        # Make prediction (probability of stroke)
        stroke_prob = model.predict_proba(input_data)[0][1]  # Probability of class 1 (stroke)
        
        # Prepare response
        response = {
            'name': data.get('name', 'User'),
            'risk_percentage': round(stroke_prob * 100, 1),
            'input_data': {
                'Age': data["Age"],
                'BMI': data["BMI"],
                'Cholesterol': data["Cholesterol"],
                'Hypertension_Category': data["Hypertension_Category"],
                'Atrial_Fibrillation': data["Atrial_Fibrillation"],
                'Diabetes': data["Diabetes"],
                'Smoking': data["Smoking"],
                'Previous_Stroke': data["Previous_Stroke"]
            },
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
