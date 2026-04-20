"""
flow_config.py
Configuration for the Cross-Asset Flow & Positioning module.

Data sources:
  - COT:          CFTC public FTP (free, official, weekly)
  - Flow Proxy:   yfinance dollar-volume (free, daily)
  - Short Interest: Nasdaq Data Link / FINRA (free tier, twice-monthly)
  - AUM:          yfinance totalAssets scrape (best-effort daily snapshot)

Output HF dataset: P2SAMAPA/p2-etf-cross-asset-flow-positioning-data
"""

import os
from datetime import datetime

# ── Hugging Face ──────────────────────────────────────────────────────────────
HF_FLOW_REPO  = "P2SAMAPA/p2-etf-cross-asset-flow-positioning-data"
HF_TOKEN      = os.environ.get("HF_TOKEN", None)

# Nasdaq Data Link (Quandl successor) — free-tier key
NASDAQ_API_KEY = os.environ.get("NASDAQ_API_KEY", None)

# ── Universe (mirrors EVT universe exactly) ───────────────────────────────────
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(dict.fromkeys(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED":       ALL_TICKERS,
}

# ── Date range ────────────────────────────────────────────────────────────────
START_DATE = "2008-01-01"
TODAY      = datetime.now().strftime("%Y-%m-%d")

# ── COT: CFTC ticker → futures market name mapping ───────────────────────────
# Used to filter the CFTC disaggregated futures CSV.
# Source: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm
COT_MARKET_MAP = {
    "GLD": "GOLD - COMMODITY EXCHANGE INC.",
    "SLV": "SILVER - COMMODITY EXCHANGE INC.",
    "TLT": "30-YEAR U.S. TREASURY BONDS - CBOT",
    "HYG": None,   # No direct COT futures — use proxy signal from TLT + credit spread
    "LQD": None,
    "VNQ": None,
    # Equity futures (used as sector proxies)
    "SPY": "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
    "QQQ": "NASDAQ-100 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
    "XLE": "LIGHT SWEET CRUDE OIL - NEW YORK MERCANTILE EXCHANGE",
    "GDX": "GOLD - COMMODITY EXCHANGE INC.",  # GDX proxy via gold futures
    "XME": "SILVER - COMMODITY EXCHANGE INC.", # Metal miner proxy via silver futures
}

# COT signal parameters
COT_LOOKBACK_WEEKS  = 52   # Rolling window for COT Index percentile (52-week norm)
COT_COLUMNS = {
    # Column names in CFTC disaggregated CSV (large spec net = longs - shorts)
    "long":  "Lev_Money_Positions_Long_All",
    "short": "Lev_Money_Positions_Short_All",
    "market":"Market_and_Exchange_Names",
    "date":  "As_of_Date_In_Form_YYMMDD",
}

# ── Flow Proxy (yfinance dollar-volume momentum) ──────────────────────────────
FLOW_PROXY_WINDOWS = [5, 21, 63]   # 1-week, 1-month, 3-month dollar-vol change
FLOW_PROXY_ZSCORE_WINDOW = 252     # z-score normalisation window

# ── Short Interest (Nasdaq Data Link / FINRA) ─────────────────────────────────
# Free tier: https://data.nasdaq.com/databases/FINRA
SHORT_INTEREST_SOURCE = "nasdaq"   # "nasdaq" or "finra_direct"
SHORT_LOOKBACK = 63                # 3-month window for short ratio momentum

# ── AUM (yfinance totalAssets) ────────────────────────────────────────────────
AUM_CHANGE_WINDOWS = [5, 21]       # 1-week, 1-month AUM change %

# ── Signal combination ────────────────────────────────────────────────────────
# Weights for composite FLOW score per ETF (normalised within universe)
SIGNAL_WEIGHTS = {
    "cot_index":          0.30,   # COT positioning percentile (contrarian/trend)
    "flow_proxy_z":       0.30,   # Dollar-volume momentum z-score
    "short_ratio_chg":    0.20,   # Short ratio momentum (contrarian)
    "aum_change":         0.20,   # AUM change % (trend-following)
}

# ── Output filenames on HF ────────────────────────────────────────────────────
HF_FILES = {
    "cot":          "cot/cot_positioning.parquet",
    "flow_proxy":   "flow_proxy/dollar_volume_flow.parquet",
    "short_interest":"short_interest/short_interest.parquet",
    "aum":          "aum/aum_history.parquet",
    "composite":    "composite/flow_composite_scores.parquet",
    "daily_json":   "daily_results/flow_positioning_{date}.json",
}

# ── Streamlit display ─────────────────────────────────────────────────────────
SIGNAL_DESCRIPTIONS = {
    "cot_index": {
        "label": "COT Index (Speculator Positioning %ile)",
        "description": "52-week percentile of leveraged-money net positioning. "
                       ">80 = crowded long (potential reversal), <20 = crowded short.",
        "high_means": "Crowded long — contrarian bearish",
        "low_means":  "Crowded short — contrarian bullish",
    },
    "flow_proxy_z": {
        "label": "Flow Proxy Z-Score (Dollar-Volume Momentum)",
        "description": "Z-score of 21-day dollar-volume change vs 252-day history. "
                       "Positive = inflow momentum, Negative = outflow.",
        "high_means": "Strong inflow momentum — trend bullish",
        "low_means":  "Outflow pressure — trend bearish",
    },
    "short_ratio_chg": {
        "label": "Short Ratio Change (3-month)",
        "description": "Change in short-interest ratio over 63 days. "
                       "Rising short ratio = bearish sentiment building.",
        "high_means": "Rising short interest — contrarian bullish setup",
        "low_means":  "Falling short interest — short covering complete",
    },
    "aum_change": {
        "label": "AUM Change % (21-day)",
        "description": "Percentage change in ETF Assets Under Management over 21 days. "
                       "Positive = fund inflows, Negative = redemptions.",
        "high_means": "Strong AUM growth — institutional inflow",
        "low_means":  "AUM declining — redemption pressure",
    },
}
