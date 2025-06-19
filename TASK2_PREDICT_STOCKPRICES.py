import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Download data
stock = "AAPL"
data = yf.download(stock, start="2024-12-01", end="2025-03-01")
data = data.dropna().copy()

# Reset index to avoid DateTimeIndex issues
data.reset_index(inplace=True)

# Preserve Date column for plotting before dropping other columns
dates_for_plot = data['Date'].copy()

# Save to CSV (optional)
data.to_csv("apple_stock_data_clean.csv", index=False)
print(" Saved as 'apple_stock_data_clean.csv'")

# Create 'next_close' as target
data['next_close'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Make sure all required columns are present and numeric
features = ['Open', 'High', 'Low', 'Volume']
data = data[features + ['next_close']]
data = data.apply(pd.to_numeric, errors='coerce')  # convert all to numbers
data.dropna(inplace=True)  # remove any rows with NaNs

# Split features and target
x = data[features]
y = data['next_close']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=32)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(Y_test, predictions))
print("RMSE:", rmse)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(Y_test.reset_index(drop=True), label='Actual Close', linewidth=2)
plt.plot(predictions, label='Predicted Close', linestyle='--')
plt.title(f"Actual vs Predicted Closing Prices (RMSE = {rmse:.2f})")

# Use saved Date column for xticks
plt.xticks(ticks=range(len(Y_test)), labels=dates_for_plot.iloc[Y_test.index].dt.strftime('%b %d'), rotation=45)

plt.xlabel("Sample")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
