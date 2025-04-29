import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from utils import load_stock_data, preprocess_data

# Settings
symbol = "AAL"  # <-- Change this to any stock you want
sequence_length = 60
train_ratio = 0.8
epochs = 20
batch_size = 32

# 1. Load data
df = load_stock_data(symbol)
print(f"Loaded {len(df)} records for {symbol}.")

# 2. Preprocess data
X, y, scaler = preprocess_data(df, feature='Close', sequence_length=sequence_length)

# 3. Split into train and test
split = int(train_ratio * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Train the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# 6. Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_real = scaler.inverse_transform(y_test)

# 7. Plot predictions
plt.figure(figsize=(12,6))
plt.plot(y_test_real, label='Real Prices')
plt.plot(predictions, label='Predicted Prices')
plt.title(f"{symbol} Stock Price Prediction")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# 8. Evaluation
mse = mean_squared_error(y_test_real, predictions)
print(f"Test Mean Squared Error: {mse:.4f}")
