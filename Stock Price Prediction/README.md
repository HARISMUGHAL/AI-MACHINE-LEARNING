# 📈 Apple Stock Price Predictor using Random Forest

This project predicts the next-day closing price of Apple Inc. (AAPL) stock using historical financial data retrieved via the Yahoo Finance API. The model is built using a Random Forest Regressor from scikit-learn.

---

## 🧠 Technologies Used

- Python 🐍
- [yfinance](https://pypi.org/project/yfinance/) – for historical stock data
- [pandas](https://pandas.pydata.org/) – for data preprocessing
- [numpy](https://numpy.org/) – for numerical operations
- [matplotlib](https://matplotlib.org/) – for plotting actual vs predicted prices
- [scikit-learn](https://scikit-learn.org/) – for model training and evaluation

---

## 📊 Features Used for Prediction

- Open price
- High price
- Low price
- Volume

**Target variable:** Next-day closing price (`next_close`)

---

## 🧪 Model Used

- **RandomForestRegressor** with `n_estimators=100`
- Performance metric: **RMSE** (Root Mean Squared Error)

---

## 📈 Visualization

- The plot shows a comparison of **actual vs predicted closing prices**.
- Date labels on the x-axis represent the sample index in the test set.

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install yfinance pandas numpy scikit-learn matplotlib
