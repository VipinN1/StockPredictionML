import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def get_available_symbols(meta_path='data/symbols_valid_meta.csv', include_etfs=False):
    df = pd.read_csv(meta_path)
    if not include_etfs:
        df = df[df['ETF'] == 'N']
    return df['Symbol'].dropna().unique().tolist()


def load_stock_data(symbol, data_folder="data/stocks"):
    """Load stock data for a given symbol"""
    filepath = os.path.join(data_folder, f"{symbol}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist.")
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

def preprocess_data(df, feature='Close', sequence_length=60):
    """Preprocess stock data to create sequences"""
    data = df[feature].values.reshape(-1, 1)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X = []
    y = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler
