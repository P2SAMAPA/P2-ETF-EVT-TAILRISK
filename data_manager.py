"""
Data loading and preprocessing for EVT engine.
Handles master_data.parquet with millisecond timestamp index.
"""

import pandas as pd
from huggingface_hub import hf_hub_download
import config

def load_master_data() -> pd.DataFrame:
    """
    Downloads master_data.parquet from Hugging Face and loads into DataFrame.
    The parquet file has a millisecond Unix timestamp as the index.
    Returns a DataFrame with columns: Date, ticker, log_return.
    """
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df)} rows from master data.")
    
    # --- Handle index as millisecond timestamp ---
    if isinstance(df.index, pd.DatetimeIndex):
        # Already datetime index
        df = df.reset_index()
        date_col = df.columns[0]
        print(f"Index is DatetimeIndex. Using column '{date_col}' as date.")
    elif df.index.dtype in ['int64', 'float64'] or df.index.name in ['timestamp', 'time', 'date']:
        # Index is numeric (milliseconds)
        print("Index appears to be numeric (milliseconds). Converting to datetime...")
        df = df.reset_index()
        timestamp_col = df.columns[0]
        # Convert milliseconds to datetime
        df['Date'] = pd.to_datetime(df[timestamp_col], unit='ms')
        df = df.drop(columns=[timestamp_col])
    else:
        # Fallback: look for date column
        possible_date_cols = ['Date', 'date', 'DATE', 'timestamp', 'time']
        date_col = None
        for col in possible_date_cols:
            if col in df.columns:
                date_col = col
                break
        if date_col is None:
            # Try to reset index and assume first column is date
            df = df.reset_index()
            date_col = df.columns[0]
            print(f"No obvious date column found. Using first column '{date_col}' as date.")
        df = df.rename(columns={date_col: 'Date'})
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # --- Detect ticker column ---
    possible_ticker_cols = ['ticker', 'Ticker', 'symbol', 'Symbol', 'asset']
    ticker_col = None
    for col in possible_ticker_cols:
        if col in df.columns:
            ticker_col = col
            break
    if ticker_col is None:
        raise KeyError("Could not find ticker column. Available columns: " + str(df.columns.tolist()))
    df = df.rename(columns={ticker_col: 'ticker'})
    
    # --- Detect log_return column ---
    possible_return_cols = ['log_return', 'Log_Return', 'log_ret', 'return', 'returns']
    return_col = None
    for col in possible_return_cols:
        if col in df.columns:
            return_col = col
            break
    if return_col is None:
        raise KeyError("Could not find log_return column. Available columns: " + str(df.columns.tolist()))
    df = df.rename(columns={return_col: 'log_return'})
    
    # Sort for rolling window consistency
    df = df.sort_values(['ticker', 'Date'])
    
    return df

def get_etf_returns(df: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Extracts log_return series for a specific ticker, indexed by Date.
    """
    etf_df = df[df['ticker'] == ticker].copy()
    etf_df = etf_df.set_index('Date')['log_return'].dropna()
    return etf_df

def get_universe_tickers(universe_name: str) -> list:
    """Returns list of tickers for a given universe."""
    return config.UNIVERSES.get(universe_name, [])
