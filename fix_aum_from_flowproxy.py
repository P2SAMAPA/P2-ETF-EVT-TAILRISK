"""
fix_aum_from_flowproxy.py
Standalone one-shot script to rebuild aum/aum_history.parquet on HF
using the flow_proxy parquet (which already has full price history)
combined with shares-outstanding from yfinance .info.

Run this once:
    python fix_aum_from_flowproxy.py

This is completely independent of flow_seed.py / flow_trainer.py.
After it succeeds, normal daily updates via flow_trainer.py will
append to the correct history.
"""

import io
import os
import time
import logging
import sys

import pandas as pd
import numpy as np
import yfinance as yf
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

HF_REPO  = "P2SAMAPA/p2-etf-cross-asset-flow-positioning-data"
HF_TOKEN = os.environ.get("HF_TOKEN")

ALL_TICKERS = [
    "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM",
]


def _pull(path: str) -> pd.DataFrame:
    try:
        local = hf_hub_download(
            repo_id=HF_REPO,
            filename=path,
            repo_type="dataset",
            token=HF_TOKEN,
            cache_dir="./hf_cache",
            force_download=True,
        )
        return pd.read_parquet(local)
    except Exception as e:
        log.warning(f"Could not pull {path}: {e}")
        return pd.DataFrame()


def _push(df: pd.DataFrame, path: str, msg: str):
    if not HF_TOKEN:
        log.warning("HF_TOKEN not set — skipping push.")
        return
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    HfApi(token=HF_TOKEN).upload_file(
        path_or_fileobj=buf,
        path_in_repo=path,
        repo_id=HF_REPO,
        repo_type="dataset",
        commit_message=msg,
    )
    log.info(f"Pushed {path} ({len(df):,} rows)")


def get_shares_outstanding() -> dict:
    """
    Fetch shares outstanding for all tickers via yfinance.
    Returns dict: {ticker: shares}. Uses multiple fallback fields.
    Works on weekends — shares_outstanding is a static field.
    """
    log.info("Fetching shares outstanding from yfinance...")
    shares_map = {}
    fields = [
        "sharesOutstanding",
        "impliedSharesOutstanding",
        "floatShares",
    ]
    for ticker in ALL_TICKERS:
        try:
            info = yf.Ticker(ticker).info
            shares = None
            for field in fields:
                v = info.get(field)
                if v and float(v) > 1e6:   # sanity: > 1M shares
                    shares = float(v)
                    break
            if shares:
                shares_map[ticker] = shares
                log.info(f"  {ticker}: {shares/1e6:.0f}M shares")
            else:
                log.warning(f"  {ticker}: shares not found in yfinance")
            time.sleep(0.3)
        except Exception as e:
            log.warning(f"  {ticker}: {e}")
    return shares_map


def build_aum_from_prices(df_flow: pd.DataFrame, shares_map: dict) -> pd.DataFrame:
    """
    AUM = Close price × shares outstanding.
    df_flow has columns: date, etf, close (from the flow_proxy parquet).
    """
    log.info("Computing AUM from price × shares...")
    frames = []
    for ticker, shares in shares_map.items():
        sub = df_flow[df_flow["etf"] == ticker][["date", "close"]].copy()
        if sub.empty:
            log.warning(f"  {ticker}: no price data in flow_proxy")
            continue
        sub["etf"] = ticker
        sub["aum"] = sub["close"] * shares
        frames.append(sub[["date", "etf", "aum"]])
        log.info(f"  {ticker}: {len(sub)} daily AUM rows, "
                 f"latest=${sub['aum'].iloc[-1]/1e9:.2f}B")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["etf", "date"]).drop_duplicates(["date", "etf"])

    # Add change signals
    result["aum_chg_5d"]  = result.groupby("etf")["aum"].pct_change(5)
    result["aum_chg_21d"] = result.groupby("etf")["aum"].pct_change(21)
    result["aum_change"]  = result["aum_chg_21d"]

    return result.reset_index(drop=True)


def main():
    log.info("=== AUM Rebuild from Flow Proxy ===")
    log.info(f"HF_TOKEN set: {bool(HF_TOKEN)}")

    # Step 1: Pull flow_proxy (has full price history 2008-present)
    log.info("\n[1/3] Pulling flow_proxy parquet from HF...")
    df_flow = _pull("flow_proxy/dollar_volume_flow.parquet")
    if df_flow.empty:
        log.error("flow_proxy parquet not found — run flow_seed.py --skip-cot --skip-short --skip-aum first")
        sys.exit(1)

    df_flow["date"] = pd.to_datetime(df_flow["date"])
    log.info(f"  Flow proxy: {len(df_flow):,} rows, "
             f"date range: {df_flow['date'].min().date()} → {df_flow['date'].max().date()}")

    # Verify 'close' column exists
    if "close" not in df_flow.columns:
        log.error(f"'close' column not in flow_proxy. Columns: {df_flow.columns.tolist()}")
        sys.exit(1)

    # Step 2: Fetch shares outstanding
    log.info("\n[2/3] Fetching shares outstanding from yfinance...")
    shares_map = get_shares_outstanding()
    if not shares_map:
        log.error("Could not get shares outstanding for any ticker.")
        sys.exit(1)
    log.info(f"  Got shares for {len(shares_map)}/{len(ALL_TICKERS)} tickers")

    # Step 3: Build and push AUM
    log.info("\n[3/3] Building AUM dataset and pushing to HF...")
    df_aum = build_aum_from_prices(df_flow, shares_map)
    if df_aum.empty:
        log.error("AUM dataset is empty.")
        sys.exit(1)

    log.info(f"\nAUM dataset summary:")
    log.info(f"  Rows:   {len(df_aum):,}")
    log.info(f"  ETFs:   {df_aum['etf'].nunique()}")
    log.info(f"  From:   {df_aum['date'].min().date()}")
    log.info(f"  To:     {df_aum['date'].max().date()}")
    log.info(f"  Size:   ~{df_aum.memory_usage(deep=True).sum()/1e6:.1f} MB in memory")

    _push(df_aum, "aum/aum_history.parquet",
          "Rebuild AUM from price×shares (flow_proxy prices 2008-present)")

    log.info("\n✅ AUM rebuild complete.")
    log.info("The composite scores now need to be recomputed.")
    log.info("Run: python flow_seed.py --skip-cot --skip-flow --skip-short --skip-aum")
    log.info("  (that will recompute composite only, using the new AUM data)")


if __name__ == "__main__":
    main()
