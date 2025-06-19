# 🏠 House Price Prediction using Gradient Boosting

This project predicts house prices using a **Gradient Boosting Regressor** trained on a real estate dataset. The goal is to accurately estimate house prices based on various features such as area, number of rooms, location, and more.

---

## 📂 Dataset Overview

- The dataset includes the following key columns:
  - `Price` – Target variable (House price)
  - `Area`, `Bedrooms`, `Bathrooms`, `Garage`, `Location`, `Condition`, `YearBuilt`
- CSV file name: `House Price Prediction Dataset.csv`

---

## 🧠 Model Overview

- **Algorithm**: Gradient Boosting Regressor (from `sklearn`)
- **Goal**: Predict `Price` (after applying log-transform)
- **Evaluation Metrics**:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score

---

## 🧹 Data Preprocessing

1. **Outlier Removal**: Top and bottom 1% values removed for key numeric columns.
2. **Feature Engineering**:
   - `Age`: Derived from `YearBuilt`.
   - `Price_per_sqft`: Price divided by area.
   - `Area_per_bedroom`: Area divided by number of bedrooms.
3. **Encoding**: One-hot encoding applied to categorical features.
4. **Log Transformation**: Log applied to the `Price` column to normalize distribution.

---

## 🧪 Training & Testing

- Data split: 80% training / 20% testing
- Parameters for `GradientBoostingRegressor`:
  - `n_estimators=1000`
  - `learning_rate=0.03`
  - `max_depth=4`
  - `min_samples_split=4`
  - `subsample=0.8`

---

## 📈 Results

Example output after training:

MAE: 12745.22
RMSE: 18560.35
R² Score: 0.9241




## 📊 Visualization

The actual vs predicted house prices are visualized using a scatter plot:

- Red dashed line = Perfect predictions
- Blue dots = Predicted values

---

## 🚀 How to Run

1. **Install dependencies**:

```bash
pip install pandas numpy matplotlib scikit-learn
Run the script:
```bash
python house_price_predictor.py
Make sure the dataset House Price Prediction Dataset.csv is in the same directory.