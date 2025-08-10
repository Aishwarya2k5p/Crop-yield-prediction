
# Crop Yield Prediction

## Overview
This project aims to predict crop yields using machine learning techniques. It consists of a Python-based web application with a simple, interactive interface for users to input relevant agricultural data and receive crop yield predictions. The system is designed to help farmers and stakeholders make informed decisions based on predictive analytics.

## Features
- Predicts crop yield based on user-input factors such as weather, soil, and crop type.
- Web application built with Flask for easy user interaction.
- Machine learning pipeline trained using real agricultural datasets.
- Data preprocessing and model serialization for consistent results.

## Tech Stack
- **Frontend:** HTML, CSS (for web UI), JavaScript
- **Backend:** Python (Flask framework)
- **Machine Learning:** scikit-learn, Pandas, pickle (for model and preprocessing storage)

## Project Structure
```
static/              # Static files (CSS, JS)
templates/           # HTML templates (UI pages)
app.py               # Main Flask backend application
final_train.py       # Model training script
check.py             # Script for evaluation or inference
mlp.pkl              # Trained ML model
preprocessor.pkl     # Data preprocessor
crop_yield.csv       # Dataset used for training and prediction
visualization.ipynb  # Jupyter notebook for data analysis and visualization
.gitignore           # Exclusions for version control
```

## Setup & Usage

1. **Clone the repository**
   ```
   git clone https://github.com/AksharaDondeti/Crop-yield-prediction.git
   cd Crop-yield-prediction
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Run the Flask application**
   ```
   python app.py
   ```
   Open your browser at `http://127.0.0.1:5000/`.

4. **Train or retrain the model** (optional)
   ```
   python final_train.py
   ```

## Dataset
The project uses `crop_yield.csv` containing historical crop and environmental data for model training and prediction.

## Visualization
Analysis notebooks and scripts (`visualization.ipynb`) allow exploration of dataset features and model predictions.

## Future Scope
While our current crop yield prediction system using a Multi-Layer Perceptron (MLP) has achieved reliable performance and successful deployment, there remains significant scope for enhancement and scalability. One promising direction involves the integration of satellite imagery and remote sensing data—such as NDVI, Sentinel-2, and Google Earth Engine (GEE)—to dynamically monitor crop health, soil moisture, and vegetation. Combining this spatial information with existing tabular inputs can significantly boost model
accuracy and enable region-specific predictions. Additionally, expanding the dataset to include more diverse
crops, regions, and agro-climatic zones from sources like ICAR, IMD, and FAO will improve the model's
generalization and adaptability, allowing it to perform well across different states or even countries with
minimal retraining. Furthermore, adopting temporal modeling techniques using LSTM or RNN architectures
can capture seasonal trends and historical rainfall patterns, leading to more robust forecasting. Incorporating
explainable AI tools such as SHAP or LIME will also enhance transparency, enabling farmers to understand
the key factors influencing predictions and thereby fostering greater trust in the system.

## License
This project is provided for academic and educational use. Refer to the repository for specifics.

---

