
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("House Price Prediction Dataset.csv")
data = data.drop(columns=["Id"])

# Remove outliers (Price, Area, Bedrooms, Bathrooms)
for col in ["Price", "Area", "Bedrooms", "Bathrooms"]:
    q1 = data[col].quantile(0.01)
    q3 = data[col].quantile(0.99)
    data = data[(data[col] >= q1) & (data[col] <= q3)]

# Feature engineering
data["Age"] = 2025 - data["YearBuilt"]
data["Price_per_sqft"] = data["Price"] / (data["Area"] + 1)
data["Area_per_bedroom"] = data["Area"] / (data["Bedrooms"] + 1)
data.drop(columns=["YearBuilt"], inplace=True)

# One-hot encode categorical features
data = pd.get_dummies(data, columns=["Location", "Condition", "Garage"], drop_first=True)

# Log-transform the price to reduce skew
data["Log_Price"] = np.log1p(data["Price"])

# Features and target
X = data.drop(columns=["Price", "Log_Price_per_sqft"], errors='ignore')
y = data["Log_Price"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boosting Model
model = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=4,
    min_samples_split=4,
    subsample=0.8,
    random_state=42
)

# Fit and predict
model.fit(x_train, y_train)
log_preds = model.predict(x_test)

# Convert back from log scale
predictions = np.expm1(log_preds)
actuals = np.expm1(y_test)

# Evaluation
mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
r2 = r2_score(actuals, predictions)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(actuals, predictions, alpha=0.6, edgecolors='w', label="Predictions")
plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2, label="Perfect Prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices.")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
