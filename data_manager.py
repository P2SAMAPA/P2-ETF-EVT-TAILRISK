"""
Data loading and preprocessing for EVT engine.
Handles master_data.parquet with DatetimeIndex or millisecond timestamp index.
Handles master_data.parquet in WIDE format (columns = tickers + macro variables).
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data() -> pd.DataFrame:
    """
    Downloads master_data.parquet from Hugging Face and loads into DataFrame.
    Returns a DataFrame with columns: Date, ticker, log_return.
    The parquet is in WIDE format: Date index (or column) + columns for each ticker/macro.
    Returns a LONG DataFrame with columns: Date, ticker, log_return.
    """
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df_wide = pd.read_parquet(file_path)
    print(f"Loaded {len(df_wide)} rows and {len(df_wide.columns)} columns.")

    # --- Step 1: Ensure 'Date' is a column (reset index if needed) ---
    if isinstance(df_wide.index, pd.DatetimeIndex):
        df_wide = df_wide.reset_index()
        date_col = df_wide.columns[0]  # Usually 'index' or 'Date'
        df_wide = df_wide.rename(columns={date_col: 'Date'})
        print(f"Reset DatetimeIndex. Date column created from '{date_col}'.")
    elif 'Date' not in df_wide.columns and 'date' not in df_wide.columns:
        # Try to use the index if it's timestamp-like
        if df_wide.index.dtype in ['int64', 'float64'] or isinstance(df_wide.index, pd.DatetimeIndex):
            df_wide = df_wide.reset_index()
            df_wide = df_wide.rename(columns={df_wide.columns[0]: 'Date'})
            df_wide['Date'] = pd.to_datetime(df_wide['Date'], unit='ms' if df_wide['Date'].dtype == 'int64' else None)
        else:
            raise KeyError("Could not locate a date column or index.")
    else:
        # Rename existing date column to 'Date'
        for col in ['Date', 'date', 'DATE', 'timestamp']:
            if col in df_wide.columns:
                df_wide = df_wide.rename(columns={col: 'Date'})
                break

    df_wide['Date'] = pd.to_datetime(df_wide['Date'])

    # --- Step 2: Identify which columns are ETFs we care about ---
    universe_tickers = set(config.FI_COMMODITIES_TICKERS + config.EQUITY_SECTORS_TICKERS)

    # Find columns that match our tickers (case-insensitive)
    ticker_columns = [col for col in df_wide.columns if col.upper() in [t.upper() for t in universe_tickers]]
    print(f"Found {len(ticker_columns)} ETF columns out of {len(universe_tickers)} expected tickers.")

    if not ticker_columns:
        raise KeyError("No ETF ticker columns found. Columns: " + str(df_wide.columns.tolist()))

    # --- Step 3: Melt only the ETF columns into long format ---
    id_vars = ['Date']
    df_long = pd.melt(
        df_wide,
        id_vars=id_vars,
        value_vars=ticker_columns,
        var_name='ticker',
        value_name='value'
    )
    
    # --- Step 4: Determine if 'value' is price or log return ---
    sample = df_long['value'].dropna().iloc[:1000]
    if sample.abs().mean() < 0.1:
        print("Values appear to be log returns (small magnitude). Using directly as 'log_return'.")
        df_long['log_return'] = df_long['value']
    else:
        print("Values appear to be prices. Computing log returns.")
        df_long = df_long.sort_values(['ticker', 'Date'])
        df_long['log_return'] = df_long.groupby('ticker')['value'].transform(lambda x: np.log(x / x.shift(1)))
    
    # Drop rows with NaN log returns (first day of each series)
    df_long = df_long.dropna(subset=['log_return'])
    
    # --- Step 5: Final cleanup ---
    df_long = df_long[['Date', 'ticker', 'log_return']].sort_values(['ticker', 'Date'])

    print(f"Final long DataFrame: {len(df_long)} rows, columns: {df_long.columns.tolist()}")
    print(f"Date range: {df_long['Date'].min()} to {df_long['Date'].max()}")

    return df_long

def get_etf_returns(df: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Extract log returns for a specific ETF ticker from the long-format DataFrame.
    """
    series = df[df['ticker'] == ticker]['log_return'].copy()
    series.index = df[df['ticker'] == ticker]['Date']
    return series.sort_index()
