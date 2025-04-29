import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from utils import load_stock_data, preprocess_data, get_available_symbols

sequence_length = 60
train_ratio = 0.8
epochs = 20
batch_size = 32

available_symbols = get_available_symbols()

print("Available Stocks:")
print(', '.join(available_symbols[:50]) + ' ...')  

symbol = input("\nEnter the stock symbol you want to analyze (e.g., AAL): ").upper()

if symbol not in available_symbols:
    print(f"❌ Symbol '{symbol}' not found in dataset. Exiting.")
    exit()

predict_future = input("Do you want to see the predicted next-day stock price? (y/n): ").lower() == 'y'

df = load_stock_data(symbol)
print(f"\nLoaded {len(df)} records for {symbol}.")

X, y, scaler = preprocess_data(df, feature='Close', sequence_length=sequence_length)

split = int(train_ratio * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_real = scaler.inverse_transform(y_test)

plt.figure(figsize=(12,6))
plt.plot(y_test_real, label='Real Prices')
plt.plot(predictions, label='Predicted Prices')
plt.title(f"{symbol} Stock Price Prediction")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

if predict_future:
    last_60_days = df['Close'].values[-sequence_length:].reshape(-1, 1)
    last_60_days_scaled = scaler.transform(last_60_days)
    last_60_days_scaled = last_60_days_scaled.reshape(1, sequence_length, 1)

    next_pred_scaled = model.predict(last_60_days_scaled)
    next_pred = scaler.inverse_transform(next_pred_scaled)

    print(f"\n Predicted next closing price for {symbol}: ${next_pred[0][0]:.2f}")


    plt.scatter(len(y_test_real), next_pred[0][0], color='red', label='Next Day Prediction')

plt.legend()
plt.show()


mse = mean_squared_error(y_test_real, predictions)
print(f"\n Test Mean Squared Error (MSE): {mse:.6f}")
