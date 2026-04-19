"""
Data loading and preprocessing for EVT engine.
"""

import pandas as pd
from huggingface_hub import hf_hub_download
import config

def load_master_data() -> pd.DataFrame:
    """
    Downloads master_data.parquet from Hugging Face and loads into DataFrame.
    Returns a DataFrame with columns: Date, ticker, log_return, (and others).
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
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
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
