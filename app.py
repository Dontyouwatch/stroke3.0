from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# Initialize model and scaler
model = None
scaler = None

def load_model_and_scaler():
    """Load the trained model and scaler from files"""
    global model, scaler
    
    try:
        model_path = Path(__file__).parent / 'models' / 'strokemodel.pkl'
        model = joblib.load(model_path)
        
        scaler_path = Path(__file__).parent / 'models' / 'scaler.pkl'
        scaler = joblib.load(scaler_path)
        
        print("Model and scaler loaded successfully")
    except Exception as e:
        print(f"Error loading model or scaler: {str(e)}")
        raise

load_model_and_scaler()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Required medical parameters
        required_fields = ['Age', 'BMI', 'Cholesterol', 'Hypertension_Category',
                         'Atrial_Fibrillation', 'Diabetes', 'Smoking', 'Previous_Stroke']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}', 'status': 'error'}), 400
        
        # Calculate derived features
        bmi = float(data["BMI"])
        bmi_cat_1 = 1 if 18.5 <= bmi < 24.9 else 0
        bmi_cat_2 = 1 if 25 <= bmi < 29.9 else 0
        bmi_cat_3 = 1 if bmi >= 30 else 0
        
        cholesterol = float(data["Cholesterol"])
        chol_cat_1 = 1 if 200 <= cholesterol < 240 else 0
        chol_cat_2 = 1 if cholesterol >= 240 else 0
        
        age = float(data["Age"])
        age_group_1 = 1 if 40 <= age < 60 else 0
        age_group_2 = 1 if age >= 60 else 0
        
        # Prepare input data
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
        
        # Scale numeric features
        numeric_features = ["Age", "Cholesterol", "BMI", "BMI_Hypertension", "Cholesterol_Atrial_Fibrillation"]
        input_data[numeric_features] = scaler.transform(input_data[numeric_features])
        
        # Ensure correct column order
        expected_columns = [
            'Age', 'BMI', 'Cholesterol', 'Hypertension_Category', 'Atrial_Fibrillation',
            'Diabetes', 'Smoking', 'Previous_Stroke', 'BMI_Category_1', 'BMI_Category_2',
            'BMI_Category_3', 'Cholesterol_Category_1', 'Cholesterol_Category_2',
            'Age_Group_1', 'Age_Group_2', 'BMI_Hypertension', 'Cholesterol_Atrial_Fibrillation'
        ]
        input_data = input_data[expected_columns]
        
        # Make prediction
        stroke_prob = model.predict_proba(input_data)[0][1]
        
        return jsonify({
            'risk_percentage': round(stroke_prob * 100, 1),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'message': 'Prediction failed'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
