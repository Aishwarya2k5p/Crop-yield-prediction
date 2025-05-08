from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Load the trained pipeline (includes preprocessing + model)
model = joblib.load("mlp.pkl")

# Define expected input columns
categorical = ["Crop", "Season", "State"]
numerical = ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Crop_Year"]
engineered = ["Fertilizer_Pesticide"]
expected_features = categorical + numerical + engineered

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction form page
@app.route("/predictor")
def predictor_page():
    return render_template("predictor.html")

# API endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        # Add engineered feature
        data["Fertilizer_Pesticide"] = float(data["Fertilizer"]) * float(data["Pesticide"])
        
        # Prepare input
        input_df = pd.DataFrame([[data.get(col, 0) for col in expected_features]], columns=expected_features)

        # Predict yield
        prediction = model.predict(input_df)[0]
        return jsonify({"predicted_yield": round(prediction, 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
