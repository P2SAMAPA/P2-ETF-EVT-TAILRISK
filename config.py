"""
Configuration for P2-ETF-EVT-TAILRISK engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"

HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-evt-tailrisk-results"

# --- Universe Definitions (mirroring master data exactly) ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]

EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]

ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- EVT Parameters ---
EVT_THRESHOLD_QUANTILE = 0.90          # Top 10% of losses considered extreme
ROLLING_WINDOW = 252                    # 1-year lookback for GPD fitting
MIN_OBSERVATIONS = 100                  # Minimum points required to fit GPD
TAIL_SHAPE_WARNING_THRESHOLD = 0.3      # ξ > 0.3 triggers warning flag
EWMA_HALFLIFE = 21                      # Smoothing for tail shape

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token (set via environment variable) ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
