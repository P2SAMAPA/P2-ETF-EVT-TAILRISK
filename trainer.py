"""
Main training / inference script for EVT Tail Risk engine.
Runs daily analysis on all universes and pushes results to Hugging Face.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

import config
import data_manager
from evt_model import EVTAnalyzer
import push_results

def run_evt_analysis():
    """Orchestrates the full EVT analysis pipeline."""
    
    print(f"=== P2-ETF-EVT-TAILRISK Run: {config.TODAY} ===")
    
    # Load master data
    df_master = data_manager.load_master_data()
    
    # Initialize analyzer
    analyzer = EVTAnalyzer(
        threshold_quantile=config.EVT_THRESHOLD_QUANTILE,
        min_obs=config.MIN_OBSERVATIONS,
        ewma_halflife=config.EWMA_HALFLIFE
    )
    
    all_results = {}
    
    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        universe_results = {}
        
        for ticker in tickers:
            print(f"  Analyzing {ticker}...")
            returns = data_manager.get_etf_returns(df_master, ticker)
            
            if len(returns) < config.MIN_OBSERVATIONS:
                print(f"    Skipping {ticker}: insufficient data ({len(returns)} obs)")
                continue
                
            evt_df = analyzer.analyze_series(returns)
            
            # Get latest values (last row)
            latest = evt_df.iloc[-1].to_dict()
            latest['ticker'] = ticker
            universe_results[ticker] = latest
            
        all_results[universe_name] = universe_results
    
    # Add metadata
    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "threshold_quantile": config.EVT_THRESHOLD_QUANTILE,
            "rolling_window": config.ROLLING_WINDOW,
            "tail_shape_warning_threshold": config.TAIL_SHAPE_WARNING_THRESHOLD,
            "ewma_halflife": config.EWMA_HALFLIFE
        },
        "universes": all_results
    }
    
    # Push to Hugging Face
    push_results.push_daily_result(output_payload)
    
    # Print summary of warnings
    print("\n=== Tail Risk Warnings (tail_shape_smooth > 0.3) ===")
    for universe_name, ticker_data in all_results.items():
        warnings = [t for t, d in ticker_data.items() if d.get('tail_warning', 0) == 1]
        if warnings:
            print(f"{universe_name}: {', '.join(warnings)}")
    
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_evt_analysis()
