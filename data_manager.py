"""
Data loading and preprocessing for EVT engine.
Handles master_data.parquet with DatetimeIndex or millisecond timestamp index.
"""

import pandas as pd
from huggingface_hub import hf_hub_download
import config

def load_master_data() -> pd.DataFrame:
    """
    Downloads master_data.parquet from Hugging Face and loads into DataFrame.
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
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Index type: {type(df.index)}, Index name: {df.index.name}")
    
    # --- Step 1: Ensure we have a 'Date' column from the index ---
    if isinstance(df.index, pd.DatetimeIndex):
        # Index is already datetime – reset and rename the new column
        df = df.reset_index()
        # The new column will be named 'index' (or the index's name if any)
        date_col = df.columns[0]  # Usually 'index' or 'Date'
        df = df.rename(columns={date_col: 'Date'})
        print(f"Reset DatetimeIndex. Date column renamed from '{date_col}' to 'Date'.")
    elif df.index.dtype in ['int64', 'float64'] or 'timestamp' in str(df.index.name).lower():
        # Index is numeric (likely milliseconds)
        df = df.reset_index()
        timestamp_col = df.columns[0]
        df['Date'] = pd.to_datetime(df[timestamp_col], unit='ms')
        df = df.drop(columns=[timestamp_col])
        print(f"Converted numeric index (ms) to datetime. Dropped '{timestamp_col}'.")
    else:
        # Try to find a date column among columns
        possible_date_cols = ['Date', 'date', 'DATE', 'timestamp', 'time']
        found = False
        for col in possible_date_cols:
            if col in df.columns:
                df = df.rename(columns={col: 'Date'})
                found = True
                print(f"Renamed existing column '{col}' to 'Date'.")
                break
        if not found:
            raise KeyError("Could not identify date column or index. Columns: " + str(df.columns.tolist()))
    
    # Now 'Date' column exists and is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # --- Step 2: Detect and rename ticker column ---
    possible_ticker_cols = ['ticker', 'Ticker', 'symbol', 'Symbol', 'asset']
    ticker_col = None
    for col in possible_ticker_cols:
        if col in df.columns:
            ticker_col = col
            break
    if ticker_col is None:
        raise KeyError("Could not find ticker column. Available columns: " + str(df.columns.tolist()))
    if ticker_col != 'ticker':
        df = df.rename(columns={ticker_col: 'ticker'})
    print(f"Using ticker column: 'ticker' (was '{ticker_col}')")
    
    # --- Step 3: Detect and rename log_return column ---
    possible_return_cols = ['log_return', 'Log_Return', 'log_ret', 'return', 'returns']
    return_col = None
    for col in possible_return_cols:
        if col in df.columns:
            return_col = col
            break
    if return_col is None:
        raise KeyError("Could not find log_return column. Available columns: " + str(df.columns.tolist()))
    if return_col != 'log_return':
        df = df.rename(columns={return_col: 'log_return'})
    print(f"Using return column: 'log_return' (was '{return_col}')")
    
    # --- Step 4: Sort for consistency ---
    df = df.sort_values(['ticker', 'Date'])
    
    # Keep only necessary columns to save memory (optional)
    # df = df[['Date', 'ticker', 'log_return']]
    
    print(f"Final columns: {df.columns.tolist()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
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
