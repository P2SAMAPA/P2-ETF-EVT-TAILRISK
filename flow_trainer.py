"""
flow_trainer.py
Daily incremental update for the Flow & Positioning module.
Called by GitHub Actions (daily_run.yml) every trading day.

What it does:
  1. Pulls existing datasets from HF
  2. Appends today's new data (COT current year refresh, new OHLCV, AUM snapshot)
  3. Recomputes composite flow scores
  4. Pushes updated parquets + dated JSON back to HF

Do NOT run this for the first time without first running:
    python flow_seed.py
"""

import json
import logging
import sys
from datetime import datetime

import pandas as pd
from huggingface_hub import HfApi

import flow_config as cfg
import flow_data_manager as dm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def push_daily_json(df_comp: pd.DataFrame):
    """
    Save today's composite scores as a dated JSON to HF.
    Mirrors EVT engine's dated-file pattern for Streamlit loading.
    """
    if df_comp is None or df_comp.empty:
        log.warning("No composite scores — skipping JSON push.")
        return

    latest_date = df_comp["date"].max()
    latest_df   = df_comp[df_comp["date"] == latest_date].copy()

    universes_out = {}
    for universe_name, tickers in cfg.UNIVERSES.items():
        universe_out = {}
        for ticker in tickers:
            row = latest_df[latest_df["etf"] == ticker]
            if row.empty:
                continue
            r = row.iloc[0]
            universe_out[ticker] = {
                "composite_score":   round(float(r.get("composite_score",   0) or 0), 4),
                "cot_index_z":       round(float(r.get("cot_index_z",       0) or 0), 4),
                "flow_proxy_z":      round(float(r.get("flow_proxy_z",      0) or 0), 4),
                "short_ratio_chg_z": round(float(r.get("short_ratio_chg_z", 0) or 0), 4),
                "aum_change_z":      round(float(r.get("aum_change_z",      0) or 0), 4),
                "flow_rank":         int(r.get("flow_rank", 0) or 0),
            }
        universes_out[universe_name] = dict(
            sorted(universe_out.items(),
                   key=lambda x: x[1]["composite_score"], reverse=True)
        )

    payload = {
        "run_date":    cfg.TODAY,
        "signal_date": str(latest_date)[:10],
        "config": {
            "signal_weights":     cfg.SIGNAL_WEIGHTS,
            "cot_lookback_weeks": cfg.COT_LOOKBACK_WEEKS,
            "flow_windows":       cfg.FLOW_PROXY_WINDOWS,
            "short_lookback":     cfg.SHORT_LOOKBACK,
        },
        "universes": universes_out,
    }

    local_file = f"flow_positioning_{cfg.TODAY}.json"
    with open(local_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info(f"Saved {local_file}")

    if not cfg.HF_TOKEN:
        log.warning("HF_TOKEN not set — skipping JSON push.")
        return

    api = HfApi(token=cfg.HF_TOKEN)
    dated_path  = cfg.HF_FILES["daily_json"].format(date=cfg.TODAY)
    latest_path = "daily_results/flow_positioning_latest.json"

    for path in [dated_path, latest_path]:
        try:
            api.upload_file(
                path_or_fileobj=local_file,
                path_in_repo=path,
                repo_id=cfg.HF_FLOW_REPO,
                repo_type="dataset",
                commit_message=f"Flow results {cfg.TODAY}",
            )
        except Exception as e:
            log.error(f"Failed to push {path}: {e}")

    log.info(f"Pushed JSON to {cfg.HF_FLOW_REPO}")


def print_summary(df_comp: pd.DataFrame):
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
        log.info(f"\n{universe_name}:")
        log.info(f"  Top:    {sub.iloc[0]['etf']}  (score={sub.iloc[0]['composite_score']:.3f})")
        log.info(f"  Bottom: {sub.iloc[-1]['etf']} (score={sub.iloc[-1]['composite_score']:.3f})")


def main():
    log.info("=" * 60)
    log.info(f"FLOW TRAINER — Daily Update: {cfg.TODAY}")
    log.info("=" * 60)

    # ── Pull existing datasets from HF ────────────────────────────────────────
    log.info("\n[1/5] Pulling existing datasets from HF...")
    existing_cot   = dm._pull_parquet(cfg.HF_FILES["cot"])
    existing_flow  = dm._pull_parquet(cfg.HF_FILES["flow_proxy"])
    existing_short = dm._pull_parquet(cfg.HF_FILES["short_interest"])
    existing_aum   = dm._pull_parquet(cfg.HF_FILES["aum"])

    log.info(f"  COT:    {'✅ ' + str(len(existing_cot)) + ' existing rows' if existing_cot is not None else '⚠️  not found'}")
    log.info(f"  Flow:   {'✅ ' + str(len(existing_flow)) + ' existing rows' if existing_flow is not None else '⚠️  not found'}")
    log.info(f"  Short:  {'✅ ' + str(len(existing_short)) + ' existing rows' if existing_short is not None else '⚠️  not found'}")
    log.info(f"  AUM:    {'✅ ' + str(len(existing_aum)) + ' existing rows' if existing_aum is not None else '⚠️  not found'}")

    if existing_cot is None and existing_flow is None and existing_aum is None:
        log.error("No seed data found on HF. Run flow_seed.py first.")
        sys.exit(1)

    # ── Step 1: Update COT (re-download current year only) ───────────────────
    log.info("\n[2/5] Updating COT (current year refresh)...")
    df_cot = dm.update_cot_incremental(existing_cot)
    if df_cot is not None and not df_cot.empty:
        dm._push_parquet(df_cot, cfg.HF_FILES["cot"], f"COT update {cfg.TODAY}")

    # ── Step 2: Update Flow Proxy (append last 5 days) ───────────────────────
    log.info("\n[3/5] Updating flow proxy (last 5 trading days)...")
    df_flow = dm.update_flow_proxy_incremental(existing_flow)
    if df_flow is not None and not df_flow.empty:
        dm._push_parquet(df_flow, cfg.HF_FILES["flow_proxy"], f"Flow proxy update {cfg.TODAY}")

    # ── Step 3: Short Interest (no daily update — keep existing) ─────────────
    # Short interest from FINRA updates twice monthly; no daily action needed.
    # The seed and periodic manual re-runs handle this.
    log.info("\n[4/5] Short interest — using existing data (updates twice monthly).")
    df_short = existing_short  # no push needed

    # ── Step 4: AUM — append today's snapshot ────────────────────────────────
    log.info("\n[4/5] Updating AUM (today's snapshot)...")
    df_aum = dm.update_aum_incremental(existing_aum)
    if df_aum is not None and not df_aum.empty:
        dm._push_parquet(df_aum, cfg.HF_FILES["aum"], f"AUM update {cfg.TODAY}")

    # ── Step 5: Recompute composite ──────────────────────────────────────────
    log.info("\n[5/5] Recomputing composite flow scores...")
    df_comp = dm.build_composite_scores(df_cot, df_flow, df_short, df_aum)
    if df_comp is not None and not df_comp.empty:
        dm._push_parquet(df_comp, cfg.HF_FILES["composite"], f"Composite update {cfg.TODAY}")

    # ── Push dated JSON ───────────────────────────────────────────────────────
    push_daily_json(df_comp)
    print_summary(df_comp)

    log.info("\n=== Flow Trainer Complete ===")


if __name__ == "__main__":
    main()
