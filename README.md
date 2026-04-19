# P2-ETF-EVT-TAILRISK

**Extreme Value Theory (Peaks-Over-Threshold) Tail Risk Engine for ETF Selection**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-EVT-TAILRISK/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-EVT-TAILRISK/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--evt--tailrisk--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-evt-tailrisk-results)
[![Streamlit Dashboard](https://img.shields.io/badge/Streamlit-Live%20Dashboard-red)](https://p2-etf-evt-tailrisk.streamlit.app)

## Overview

`P2-ETF-EVT-TAILRISK` is a quantitative engine that estimates **tail risk** across the P2Quant ETF universe using Extreme Value Theory (EVT). Unlike traditional risk models that focus on average behavior or correlation, this engine specializes in modeling **extreme losses**—the fat tails that cause portfolio blow‑ups.

The engine fits a **Generalized Pareto Distribution (GPD)** to the top 10% of historical losses (Peaks‑Over‑Threshold method) and produces daily metrics including:

- **Tail Shape (ξ)** – Measures heaviness of the left tail (ξ > 0.3 triggers a warning)
- **Value‑at‑Risk (VaR 99%)** – 1‑day loss threshold at 99% confidence
- **Expected Shortfall (ES 99%)** – Average loss *if* VaR is breached
- **Tail Warning Flag** – Binary indicator when tail risk is elevated

Results are pushed daily to a dedicated Hugging Face dataset and visualized via a professional Streamlit dashboard.

## Universe Coverage

The engine analyzes **20 ETFs** across three universes:

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

Data is sourced from the unified master dataset:  
[`P2SAMAPA/fi-etf-macro-signal-master-data`](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data) (`master_data.parquet`)

## Methodology

### Peaks‑Over‑Threshold (POT) with GPD

1. **Threshold Selection:** The 90th percentile of rolling 1‑year losses defines the extreme threshold *u*.
2. **Exceedance Modeling:** Losses beyond *u* are modeled with a Generalized Pareto Distribution (GPD).
3. **Parameter Estimation:** Shape (ξ) and scale (σ) are estimated via Maximum Likelihood (SciPy).
4. **Risk Metrics:** VaR and ES are derived from the fitted GPD using the conditional EVT framework of McNeil & Frey (2000).
5. **Smoothing & Warning:** Tail shape is smoothed with an EWMA (halflife=21 days). A `tail_warning` flag is raised when smoothed ξ > 0.3.

### Computational Efficiency

The entire backtest (20 ETFs × ~4,000 days) runs in **< 2 minutes** on a single CPU core, making it ideal for GitHub Actions free tier.

## File Structure
P2-ETF-EVT-TAILRISK/
├── config.py # All paths, universes, EVT parameters
├── data_manager.py # Data loading from Hugging Face
├── evt_model.py # Core EVT logic (GPD fitting, VaR/ES)
├── trainer.py # Main orchestration script
├── push_results.py # Upload results to Hugging Face dataset
├── streamlit_app.py # Interactive dashboard
├── requirements.txt # Python dependencies
├── .github/workflows/
│ └── daily_run.yml # Scheduled GitHub Action
└── .streamlit/
└── config.toml # Streamlit theme

text

## Configuration

All tunable parameters are in `config.py`:

```python
EVT_THRESHOLD_QUANTILE = 0.90       # Extreme loss threshold (10% tail)
ROLLING_WINDOW = 252                # 1‑year lookback
TAIL_SHAPE_WARNING_THRESHOLD = 0.3  # ξ threshold for warning flag
EWMA_HALFLIFE = 21                  # Smoothing factor
Running Locally
Clone the repository

bash
git clone https://github.com/P2SAMAPA/P2-ETF-EVT-TAILRISK.git
cd P2-ETF-EVT-TAILRISK
Install dependencies

bash
pip install -r requirements.txt
Set Hugging Face token (optional, for pushing results)

bash
export HF_TOKEN="your_hf_token_here"
Run the engine

bash
python trainer.py
Launch the dashboard

bash
streamlit run streamlit_app.py
GitHub Actions Automation
The engine runs automatically Monday–Friday at 10:00 UTC via .github/workflows/daily_run.yml.
To enable:

Add HF_TOKEN as a repository secret (Settings → Secrets and variables → Actions)

The workflow can also be triggered manually via the Actions tab.

Results & Dashboard
Daily results are stored as JSON files in:
P2SAMAPA/p2-etf-evt-tailrisk-results

Each file follows the naming convention: evt_tailrisk_YYYY-MM-DD.json

Dashboard Features
Current Tail Risk Summary – Sortable table with color‑coded warnings

Historical Tail Shape Analysis – Interactive time‑series of ξ for any ETF

Universe Heatmap – Bar chart ranking all ETFs by current tail risk

Dark Mode Support – Professional styling with custom CSS

Integration with Other P2Quant Engines
This engine complements the existing suite by providing a tail‑risk gate.
Example use in a meta‑controller:

python
if evt_results[ticker]['tail_warning'] == 1:
    # Override any BUY signal from other engines
    position = "CASH"
Dependencies
pandas, numpy, scipy – Core data and statistical modeling

huggingface_hub – Dataset access

streamlit, plotly – Dashboard visualization

pyarrow – Parquet support

License
MIT License – see LICENSE file for details.

Maintainer: P2SAMAPA
Last Updated: April 2026
