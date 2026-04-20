"""
flow_seed.py
ONE-TIME historical data seeding script (2008 → present).

Run this ONCE manually (or via a manual workflow_dispatch trigger) to
populate the HF dataset P2SAMAPA/p2-etf-cross-asset-flow-positioning-data
before the daily flow_trainer.py starts running.

Usage:
    python flow_seed.py
    python flow_seed.py --skip-cot        # skip COT if already seeded
    python flow_seed.py --skip-flow       # skip flow proxy if already seeded
    python flow_seed.py --skip-short      # skip short interest
    python flow_seed.py --skip-aum        # skip AUM (only snapshot anyway)

After running this, use flow_trainer.py for all future daily updates.
Do NOT run flow_seed.py again unless you need a complete fresh rebuild.
"""

import argparse
import logging
import sys

import flow_config as cfg
import flow_data_manager as dm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("flow_seed.log"),
    ],
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="One-time historical seeding for the Flow & Positioning module."
    )
    parser.add_argument("--skip-cot",   action="store_true", help="Skip COT download")
    parser.add_argument("--skip-flow",  action="store_true", help="Skip flow proxy download")
    parser.add_argument("--skip-short", action="store_true", help="Skip short interest download")
    parser.add_argument("--skip-aum",   action="store_true", help="Skip AUM snapshot")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("FLOW MODULE — HISTORICAL SEED")
    log.info(f"Date range: {cfg.START_DATE} → {cfg.TODAY}")
    log.info(f"Output HF repo: {cfg.HF_FLOW_REPO}")
    log.info(f"HF_TOKEN set: {bool(cfg.HF_TOKEN)}")
    log.info(f"NASDAQ_API_KEY set: {bool(cfg.NASDAQ_API_KEY)}")
    log.info("=" * 60)

    # ── Step 1: COT ───────────────────────────────────────────────────────────
    df_cot = None
    if not args.skip_cot:
        log.info("\n[1/5] Downloading CFTC COT data (2008–present)...")
        log.info("  Downloading ~17 annual zip files from www.cftc.gov")
        df_cot = dm.build_cot_dataset(start_year=2008)
        if df_cot is not None and not df_cot.empty:
            dm._push_parquet(df_cot, cfg.HF_FILES["cot"], "Seed: COT positioning 2008-present")
            log.info(f"  ✅ COT seeded: {len(df_cot):,} rows, {df_cot['etf'].nunique()} ETFs")
        else:
            log.warning("  ⚠️  COT returned no data — check CFTC availability")
    else:
        log.info("[1/5] Skipping COT (--skip-cot)")
        df_cot = dm._pull_parquet(cfg.HF_FILES["cot"])

    # ── Step 2: Flow Proxy ────────────────────────────────────────────────────
    df_flow = None
    if not args.skip_flow:
        log.info(f"\n[2/5] Downloading yfinance OHLCV for {len(cfg.ALL_TICKERS)} ETFs (2008–present)...")
        df_flow = dm.build_flow_proxy_dataset()
        if df_flow is not None and not df_flow.empty:
            dm._push_parquet(df_flow, cfg.HF_FILES["flow_proxy"], "Seed: Flow proxy 2008-present")
            log.info(f"  ✅ Flow proxy seeded: {len(df_flow):,} rows")
        else:
            log.warning("  ⚠️  Flow proxy returned no data")
    else:
        log.info("[2/5] Skipping flow proxy (--skip-flow)")
        df_flow = dm._pull_parquet(cfg.HF_FILES["flow_proxy"])

    # ── Step 3: Short Interest ────────────────────────────────────────────────
    df_short = None
    if not args.skip_short:
        key_info = ("Nasdaq Data Link" if cfg.NASDAQ_API_KEY
                    else "FINRA direct API (no key — ~2yr history only)")
        log.info(f"\n[3/5] Downloading short interest via {key_info}...")
        if not cfg.NASDAQ_API_KEY:
            log.info("  Tip: set NASDAQ_API_KEY for longer short interest history")
        df_short = dm.build_short_interest_dataset()
        if df_short is not None and not df_short.empty:
            dm._push_parquet(df_short, cfg.HF_FILES["short_interest"],
                             "Seed: Short interest")
            log.info(f"  ✅ Short interest seeded: {len(df_short):,} rows")
        else:
            log.warning("  ⚠️  Short interest returned no data")
    else:
        log.info("[3/5] Skipping short interest (--skip-short)")
        df_short = dm._pull_parquet(cfg.HF_FILES["short_interest"])

    # ── Step 4: AUM ───────────────────────────────────────────────────────────
    df_aum = None
    if not args.skip_aum:
        log.info("\n[4/5] Fetching AUM snapshots via yfinance...")
        log.info("  Note: yfinance only provides current snapshot.")
        log.info("  AUM history will grow as flow_trainer.py runs daily.")
        df_aum = dm.build_aum_dataset()
        if df_aum is not None and not df_aum.empty:
            dm._push_parquet(df_aum, cfg.HF_FILES["aum"], "Seed: AUM initial snapshot")
            log.info(f"  ✅ AUM seeded: {len(df_aum)} rows ({df_aum['etf'].nunique()} ETFs)")
        else:
            log.warning("  ⚠️  AUM returned no data")
    else:
        log.info("[4/5] Skipping AUM (--skip-aum)")
        df_aum = dm._pull_parquet(cfg.HF_FILES["aum"])

    # ── Step 5: Composite scores ──────────────────────────────────────────────
    log.info("\n[5/5] Building composite flow scores...")
    df_comp = dm.build_composite_scores(df_cot, df_flow, df_short, df_aum)
    if df_comp is not None and not df_comp.empty:
        dm._push_parquet(df_comp, cfg.HF_FILES["composite"],
                         "Seed: Composite flow scores")
        log.info(f"  ✅ Composite seeded: {len(df_comp):,} rows")
    else:
        log.warning("  ⚠️  Composite scores empty — need at least flow proxy data")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("SEED COMPLETE — Summary:")
    log.info(f"  COT:            {'✅ ' + str(len(df_cot)) + ' rows' if df_cot is not None and not df_cot.empty else '⚠️  empty/skipped'}")
    log.info(f"  Flow Proxy:     {'✅ ' + str(len(df_flow)) + ' rows' if df_flow is not None and not df_flow.empty else '⚠️  empty/skipped'}")
    log.info(f"  Short Interest: {'✅ ' + str(len(df_short)) + ' rows' if df_short is not None and not df_short.empty else '⚠️  empty/skipped'}")
    log.info(f"  AUM:            {'✅ ' + str(len(df_aum)) + ' rows' if df_aum is not None and not df_aum.empty else '⚠️  empty/skipped'}")
    log.info(f"  Composite:      {'✅ ' + str(len(df_comp)) + ' rows' if df_comp is not None and not df_comp.empty else '⚠️  empty'}")
    log.info("=" * 60)
    log.info("Next: commit all flow_*.py files, then let daily_run.yml")
    log.info("handle all future updates via flow_trainer.py")


if __name__ == "__main__":
    main()
