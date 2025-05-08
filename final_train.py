import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import loguniform

import warnings
warnings.filterwarnings("ignore")

print("🔄 Loading dataset...")

# Load dataset
df = pd.read_csv("crop_yield.csv")
print(f"✅ Original dataset shape: {df.shape}")

# --- Define categorical and numerical columns ---
categorical = ["Crop", "Season", "State"]
numerical = ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Crop_Year"]

# --- Handle Missing Values ---
df[numerical] = df[numerical].fillna(df[numerical].median())
for col in categorical:
    df[col] = df[col].fillna(df[col].mode()[0])

print(f"📊 After filling missing values: {df.shape}")

# --- Filter invalid values ---
df = df[df["Yield"] > 0]
df = df[df["Area"] > 0]
print(f"📉 After removing invalid Yield/Area: {df.shape}")

# --- Mild Outlier Removal ---
def remove_outliers(df, col, factor=2.0):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

for col in ["Yield", "Area"]:
    df = remove_outliers(df, col)

print(f"🧹 After mild outlier removal: {df.shape}")

# --- Feature Engineering ---
df['Fertilizer_Pesticide'] = df['Fertilizer'] * df['Pesticide']

# --- Features and Labels ---
X = df.drop(columns=["Yield", "Production"])
y = df["Yield"]

# --- Preprocessing and Model Pipeline ---
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numerical)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", MLPRegressor(
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=42
    ))
])

# --- Hyperparameter Tuning using RandomizedSearchCV (Faster) ---
param_dist = {
    "regressor__hidden_layer_sizes": [(100,), (150,), (100, 50), (200, 100)],
    "regressor__activation": ['relu', 'tanh'],
    "regressor__solver": ['adam'],
    "regressor__learning_rate": ['constant', 'adaptive'],
    "regressor__alpha": loguniform(1e-4, 1e-2)  # Regularization: sampled log-uniformly
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=20,  # Try 20 random combinations
    cv=3,       # Use 3-fold to speed up
    n_jobs=-1,
    verbose=3,
    scoring='neg_mean_absolute_error',
    random_state=42
)

print("🚀 Starting hyperparameter search...")
random_search.fit(X, y)

# --- Best Hyperparameters ---
print(f"\n✅ Best Hyperparameters: {random_search.best_params_}")

# --- Model Evaluation ---
best_model = random_search.best_estimator_

# --- Cross-Validation Scores ---
cross_val_scores = cross_val_score(best_model, X, y, cv=3, scoring='neg_mean_absolute_error')
print(f"Cross-validation MAE scores: {cross_val_scores}")
print(f"Average Cross-validation MAE: {-cross_val_scores.mean():.2f}")

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the best model on the training data
best_model.fit(X_train, y_train)

# Predictions and Evaluation on Test Data
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n📈 Evaluation Results:")
print(f" - Mean Absolute Error      : {mae:.2f}")
print(f" - Mean Squared Error       : {mse:.2f}")
print(f" - Root Mean Squared Error  : {rmse:.2f}")
print(f" - R-squared                : {r2:.2f}")

# --- Save Trained Model ---
joblib.dump(best_model, "mlp.pkl")
print("💾 Model saved as 'mlp.pkl'")

