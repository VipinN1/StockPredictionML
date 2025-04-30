import glob
import pandas as pd
import numpy as np
from pathlib import Path

def load_data(path="data/stocks", cutoff_date=None):
    """
    Reads every CSV under `path` into a single DataFrame,
    normalizes column names to snake_case, adds a `symbol` column,
    and optionally filters out rows before `cutoff_date`.
    """
    files = glob.glob(str(Path(path) / "*.csv"))
    df_list = []
    for fp in files:
        sym = Path(fp).stem
        tmp = pd.read_csv(fp, parse_dates=["Date"])
        # normalize column names in each file
        tmp.columns = (
            tmp.columns
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^\w_]', '', regex=True)
        )
        tmp['symbol'] = sym
        df_list.append(tmp)
    df = pd.concat(df_list, ignore_index=True)
    # normalize all columns again and fix any leftover names
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[^\w_]', '', regex=True)
    )
    # explicit rename in case 'adj close' persisted
    if 'adj_close' not in df.columns and 'adj_close' in df.columns:
        df = df.rename(columns={'adj close': 'adj_close'})
    if cutoff_date is not None:
        df = df[df.date >= pd.to_datetime(cutoff_date)]
    return df


def make_features(df, symbol, ma_window=10):
    """
    Filters `df` to one ticker, computes:
      - log_return (logarithmic daily return)
      - moving average over `ma_window` days of adj_close
      - next-day log_return as `target`
    Returns cleaned DataFrame with columns ['log_return', 'ma{window}', 'volume', 'target'].
    """
    # ensure df columns normalized
    df = df.copy()
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[^\w_]', '', regex=True)
    )
    sub = df[df.symbol == symbol].copy()
    sub = sub.set_index('date').sort_index()

    # ensure we have adj_close
    if 'adj_close' not in sub.columns:
        raise KeyError("Column 'adj_close' not found in data. Have columns: {}".format(sub.columns.tolist()))

    sub['log_return'] = np.log(sub['adj_close']) - np.log(sub['adj_close'].shift(1))
    sub[f'ma{ma_window}'] = sub['adj_close'].rolling(ma_window).mean()
    sub['target'] = sub['log_return'].shift(-1)
    sub = sub.dropna()
    cols = ['log_return', f'ma{ma_window}', 'volume', 'target']
    return sub[cols]


def make_windows(x: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Converts arrays X (N×D) and y (N×1) into:
      - X_windows: (N-seq_len × seq_len × D)
      - y_windows: (N-seq_len × 1)
    where seq_len is the number of historical steps.
    """
    Xw, yw = [], []
    for i in range(len(x) - seq_len):
        Xw.append(x[i : i + seq_len])
        yw.append(y[i + seq_len])
    return np.stack(Xw), np.vstack(yw)
