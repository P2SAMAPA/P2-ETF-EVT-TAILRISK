"""
flow_trainer.py
Orchestrator for the Cross-Asset Flow & Positioning module.
Called daily by GitHub Actions alongside the EVT trainer.

Usage:
  python flow_trainer.py            # daily incremental update
  python flow_trainer.py --backfill # first-time full historical load
"""

import argparse
import json
import logging
import os
from datetime import datetime

import pandas as pd
from huggingface_hub import HfApi

import flow_config as cfg
import flow_data_manager as dm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def push_daily_json(df_comp: pd.DataFrame):
    """
    Save today's composite scores as a dated JSON file on HF.
    Mirrors the EVT engine's dated-file pattern so the Streamlit
    dashboard can load both modules from the same repo.
    """
    if df_comp is None or df_comp.empty:
        log.warning("No composite scores — skipping JSON push.")
        return

    today_str = cfg.TODAY

    # Latest available date in composite (may be T-1 for short interest lag)
    latest_date = df_comp["date"].max()
    latest_df = df_comp[df_comp["date"] == latest_date].copy()

    # Build universe-keyed payload matching EVT engine structure
    universes_out = {}
    for universe_name, tickers in cfg.UNIVERSES.items():
        universe_out = {}
        for ticker in tickers:
            row = latest_df[latest_df["etf"] == ticker]
            if row.empty:
                continue
            r = row.iloc[0]
            universe_out[ticker] = {
                "composite_score":    round(float(r.get("composite_score",    0)), 4),
                "cot_index_z":        round(float(r.get("cot_index_z",        0)), 4),
                "flow_proxy_z":       round(float(r.get("flow_proxy_z",       0)), 4),
                "short_ratio_chg_z":  round(float(r.get("short_ratio_chg_z",  0)), 4),
                "aum_change_z":       round(float(r.get("aum_change_z",       0)), 4),
                "flow_rank":          int(r.get("flow_rank", 0)),
            }
        # Sort by composite score descending
        universes_out[universe_name] = dict(
            sorted(universe_out.items(),
                   key=lambda x: x[1]["composite_score"], reverse=True)
        )

    payload = {
        "run_date":    today_str,
        "signal_date": str(latest_date)[:10],
        "config": {
            "signal_weights":     cfg.SIGNAL_WEIGHTS,
            "cot_lookback_weeks": cfg.COT_LOOKBACK_WEEKS,
            "flow_windows":       cfg.FLOW_PROXY_WINDOWS,
            "short_lookback":     cfg.SHORT_LOOKBACK,
        },
        "universes": universes_out,
    }

    filename = cfg.HF_FILES["daily_json"].format(date=today_str)
    local_filename = f"flow_positioning_{today_str}.json"

    with open(local_filename, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info(f"Saved {local_filename}")

    if cfg.HF_TOKEN:
        api = HfApi(token=cfg.HF_TOKEN)
        api.upload_file(
            path_or_fileobj=local_filename,
            path_in_repo=filename,
            repo_id=cfg.HF_FLOW_REPO,
            repo_type="dataset",
            commit_message=f"Flow positioning results {today_str}",
        )
        # Also upload as "latest" for easy Streamlit access
        api.upload_file(
            path_or_fileobj=local_filename,
            path_in_repo="daily_results/flow_positioning_latest.json",
            repo_id=cfg.HF_FLOW_REPO,
            repo_type="dataset",
            commit_message=f"Flow positioning latest {today_str}",
        )
        log.info(f"Pushed JSON to {cfg.HF_FLOW_REPO}")


def print_summary(df_comp: pd.DataFrame):
    """Print a readable signal summary to the Actions log."""
    if df_comp is None or df_comp.empty:
        return
    latest = df_comp[df_comp["date"] == df_comp["date"].max()]
    log.info("\n=== FLOW SIGNAL SUMMARY ===")
    log.info(f"Signal date: {df_comp['date'].max():%Y-%m-%d}")
    for universe_name, tickers in cfg.UNIVERSES.items():
        sub = latest[latest["etf"].isin(tickers)].sort_values(
            "composite_score", ascending=False
        )
        if sub.empty:
            continue
        top    = sub.iloc[0]["etf"]
        bottom = sub.iloc[-1]["etf"]
        log.info(f"\n{universe_name}:")
        log.info(f"  Top    flow signal: {top}    "
                 f"(score={sub.iloc[0]['composite_score']:.3f})")
        log.info(f"  Bottom flow signal: {bottom} "
                 f"(score={sub.iloc[-1]['composite_score']:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Flow & Positioning Trainer")
    parser.add_argument("--backfill", action="store_true",
                        help="Run full historical backfill from 2008 (first run only)")
    args = parser.parse_args()

    log.info(f"=== Flow & Positioning Module: {cfg.TODAY} ===")
    log.info(f"Mode: {'BACKFILL' if args.backfill else 'DAILY UPDATE'}")

    if args.backfill:
        df_cot, df_flow, df_short, df_aum, df_comp = dm.run_full_backfill()
    else:
        df_cot, df_flow, df_short, df_aum, df_comp = dm.run_daily_update()

    push_daily_json(df_comp)
    print_summary(df_comp)
    log.info("=== Flow Module Complete ===")


if __name__ == "__main__":
    main()
