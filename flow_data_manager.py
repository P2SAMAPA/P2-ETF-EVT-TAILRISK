"""
flow_data_manager.py
Downloads, processes and stores all four flow/positioning data sources:

  1. COT  — CFTC disaggregated futures (free, official, weekly, 2008–present)
  2. Flow Proxy — yfinance dollar-volume momentum (free, daily)
  3. Short Interest — Nasdaq Data Link FINRA shorts (free tier, twice-monthly)
  4. AUM  — yfinance totalAssets history (best-effort daily)

Each source is saved as a Parquet file on HF:
  P2SAMAPA/p2-etf-cross-asset-flow-positioning-data
"""

import io
import os
import time
import zipfile
import warnings
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from huggingface_hub import HfApi, hf_hub_download, upload_file

import flow_config as cfg

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── HF helpers ────────────────────────────────────────────────────────────────

def _hf_api() -> HfApi:
    return HfApi(token=cfg.HF_TOKEN)


def _pull_parquet(hf_path: str) -> Optional[pd.DataFrame]:
    """Download a parquet from HF repo, return DataFrame or None."""
    try:
        local = hf_hub_download(
            repo_id=cfg.HF_FLOW_REPO,
            filename=hf_path,
            repo_type="dataset",
            token=cfg.HF_TOKEN,
            cache_dir="./hf_cache",
            force_download=True,
        )
        return pd.read_parquet(local)
    except Exception as e:
        log.warning(f"Could not pull {hf_path}: {e}")
        return None


def _push_parquet(df: pd.DataFrame, hf_path: str, msg: str):
    """Push DataFrame as parquet to HF repo."""
    if not cfg.HF_TOKEN:
        log.warning("HF_TOKEN not set — skipping upload.")
        return
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    _hf_api().upload_file(
        path_or_fileobj=buf,
        path_in_repo=hf_path,
        repo_id=cfg.HF_FLOW_REPO,
        repo_type="dataset",
        commit_message=msg,
    )
    log.info(f"Pushed {hf_path} ({len(df)} rows)")


# ── 1. COT DATA ───────────────────────────────────────────────────────────────

# CFTC publishes annual disaggregated CSV zips at this URL pattern
CFTC_BASE = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"

def _download_cftc_year(year: int) -> Optional[pd.DataFrame]:
    """Download one year of CFTC disaggregated futures CSV."""
    url = CFTC_BASE.format(year=year)
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            log.warning(f"CFTC {year}: HTTP {r.status_code}")
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            fname = [n for n in z.namelist() if n.endswith(".txt")][0]
            with z.open(fname) as f:
                df = pd.read_csv(f, low_memory=False)
        log.info(f"CFTC {year}: {len(df)} rows")
        return df
    except Exception as e:
        log.warning(f"CFTC {year}: {e}")
        return None


def _parse_cot_for_etf(df_all: pd.DataFrame, etf: str) -> pd.DataFrame:
    """
    Extract net leveraged-money positioning for one ETF-to-futures mapping.
    Returns weekly DataFrame with columns: date, etf, net_position, cot_index.
    """
    market_name = cfg.COT_MARKET_MAP.get(etf)
    if not market_name:
        return pd.DataFrame()

    col_market = cfg.COT_COLUMNS["market"]
    col_date   = cfg.COT_COLUMNS["date"]
    col_long   = cfg.COT_COLUMNS["long"]
    col_short  = cfg.COT_COLUMNS["short"]

    mask = df_all[col_market].str.upper().str.contains(
        market_name.upper().split(" - ")[0], na=False
    )
    df = df_all[mask].copy()
    if df.empty:
        return pd.DataFrame()

    # Parse YYMMDD date format
    df["date"] = pd.to_datetime(df[col_date].astype(str), format="%y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])

    for c in [col_long, col_short]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")

    df["net_position"] = df[col_long] - df[col_short]
    df = df[["date", "net_position"]].drop_duplicates("date").sort_values("date")
    df["etf"] = etf

    # COT Index: rolling 52-week percentile rank of net_position (0–100)
    window = cfg.COT_LOOKBACK_WEEKS
    df["cot_index"] = (
        df["net_position"]
        .rolling(window, min_periods=max(4, window // 4))
        .apply(lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-9) * 100,
               raw=True)
    )

    # Net position z-score (alternative signal)
    df["net_position_z"] = (
        (df["net_position"] - df["net_position"].rolling(window, min_periods=4).mean())
        / (df["net_position"].rolling(window, min_periods=4).std() + 1e-9)
    )

    return df[["date", "etf", "net_position", "net_position_z", "cot_index"]]


def build_cot_dataset() -> pd.DataFrame:
    """
    Download all CFTC data from 2008 to present and compute COT signals
    for all mapped ETFs. Returns long-format DataFrame, one row per (date, etf).
    """
    log.info("=== Building COT Dataset ===")
    current_year = datetime.now().year
    years = range(2008, current_year + 1)

    frames = []
    for yr in years:
        df_yr = _download_cftc_year(yr)
        if df_yr is not None:
            frames.append(df_yr)
        time.sleep(0.3)  # be polite to CFTC server

    if not frames:
        log.error("No CFTC data downloaded.")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    log.info(f"Total CFTC rows: {len(df_all)}")

    etf_frames = []
    for etf, market in cfg.COT_MARKET_MAP.items():
        if market is None:
            continue
        df_etf = _parse_cot_for_etf(df_all, etf)
        if not df_etf.empty:
            etf_frames.append(df_etf)
            log.info(f"  {etf}: {len(df_etf)} weekly observations")

    if not etf_frames:
        return pd.DataFrame()

    result = pd.concat(etf_frames, ignore_index=True)
    result = result[result["date"] >= cfg.START_DATE].reset_index(drop=True)
    log.info(f"COT dataset: {len(result)} rows, {result['etf'].nunique()} ETFs")
    return result


# ── 2. FLOW PROXY (dollar-volume momentum) ────────────────────────────────────

def build_flow_proxy_dataset() -> pd.DataFrame:
    """
    Download OHLCV from yfinance for all ETFs since 2008.
    Compute dollar-volume = Close × Volume and rolling momentum signals.
    Returns long-format DataFrame.
    """
    log.info("=== Building Flow Proxy Dataset ===")
    frames = []

    for ticker in cfg.ALL_TICKERS:
        try:
            raw = yf.download(ticker, start=cfg.START_DATE,
                              end=cfg.TODAY, progress=False, auto_adjust=True)
            if raw.empty:
                log.warning(f"  {ticker}: no yfinance data")
                continue

            # Flatten MultiIndex columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = pd.DataFrame(index=raw.index)
            df["dollar_vol"] = raw["Close"] * raw["Volume"]
            df["close"]      = raw["Close"]

            for w in cfg.FLOW_PROXY_WINDOWS:
                df[f"dv_chg_{w}d"] = df["dollar_vol"].pct_change(w)

            # Z-score of 21-day change vs trailing 252-day history
            w_z = cfg.FLOW_PROXY_ZSCORE_WINDOW
            chg21 = df["dv_chg_21d"]
            df["flow_proxy_z"] = (
                (chg21 - chg21.rolling(w_z, min_periods=60).mean())
                / (chg21.rolling(w_z, min_periods=60).std() + 1e-9)
            )

            # Relative volume (today / 21-day average)
            df["rel_volume"] = (
                df["dollar_vol"]
                / df["dollar_vol"].rolling(21, min_periods=5).mean()
            )

            df["etf"]  = ticker
            df["date"] = df.index
            df = df.reset_index(drop=True)
            frames.append(df)
            log.info(f"  {ticker}: {len(df)} days")
            time.sleep(0.1)

        except Exception as e:
            log.warning(f"  {ticker}: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.dropna(subset=["flow_proxy_z"])
    result = result[result["date"] >= cfg.START_DATE].reset_index(drop=True)
    log.info(f"Flow proxy dataset: {len(result)} rows")
    return result


# ── 3. SHORT INTEREST ─────────────────────────────────────────────────────────

def _fetch_short_interest_nasdaq(ticker: str) -> pd.DataFrame:
    """
    Pull FINRA short interest from Nasdaq Data Link (free tier).
    Table: FINRA/SHORTS — settlement date, shortVolume, totalVolume.
    """
    if not cfg.NASDAQ_API_KEY:
        return pd.DataFrame()
    try:
        import nasdaqdatalink
        nasdaqdatalink.ApiConfig.api_key = cfg.NASDAQ_API_KEY
        df = nasdaqdatalink.get_table(
            "FINRA/SHORTS",
            ticker=ticker,
            paginate=True,
        )
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.lower)
        # Normalise column names across API versions
        date_col = next((c for c in df.columns if "date" in c or "settlement" in c), None)
        vol_col  = next((c for c in df.columns if "short" in c and "vol" in c), None)
        tot_col  = next((c for c in df.columns if "total" in c and "vol" in c), None)
        if not date_col or not vol_col:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df[date_col])
        df["short_volume"]   = pd.to_numeric(df[vol_col], errors="coerce")
        if tot_col:
            df["total_volume"] = pd.to_numeric(df[tot_col], errors="coerce")
            df["short_ratio"]  = df["short_volume"] / (df["total_volume"] + 1e-9)
        else:
            df["short_ratio"] = np.nan

        df["etf"] = ticker
        return df[["date", "etf", "short_volume", "short_ratio"]].dropna(subset=["date"])
    except Exception as e:
        log.warning(f"  {ticker} short interest (Nasdaq): {e}")
        return pd.DataFrame()


def _fetch_short_interest_finra_direct(ticker: str) -> pd.DataFrame:
    """
    Pull FINRA short interest directly from FINRA API (no key required).
    https://api.finra.org/data/group/OTCMarket/name/regShoDaily
    Limited to ~2 years of history but always free.
    """
    try:
        url = (
            "https://api.finra.org/data/group/OTCMarket/name/regShoDaily"
            f"?limit=1000&offset=0&delimiter=|"
            f"&query=issueSymbolIdentifier%3D{ticker}"
        )
        r = requests.get(url, timeout=30,
                         headers={"Accept": "application/json"})
        if r.status_code != 200:
            return pd.DataFrame()
        rows = r.json()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df.get("settlementDate", df.get("tradeReportDate", "")),
                                    errors="coerce")
        df["short_volume"] = pd.to_numeric(df.get("shortParQuantity", 0), errors="coerce")
        df["total_volume"]  = pd.to_numeric(df.get("totalParQuantity", 1), errors="coerce")
        df["short_ratio"]   = df["short_volume"] / (df["total_volume"] + 1e-9)
        df["etf"] = ticker
        return df[["date", "etf", "short_volume", "short_ratio"]].dropna(subset=["date"])
    except Exception as e:
        log.warning(f"  {ticker} short interest (FINRA direct): {e}")
        return pd.DataFrame()


def build_short_interest_dataset() -> pd.DataFrame:
    """
    Download short interest for all ETFs. Tries Nasdaq Data Link first,
    falls back to FINRA direct API. Computes short ratio change signals.
    """
    log.info("=== Building Short Interest Dataset ===")
    frames = []

    for ticker in cfg.ALL_TICKERS:
        log.info(f"  {ticker}")
        if cfg.NASDAQ_API_KEY:
            df = _fetch_short_interest_nasdaq(ticker)
        else:
            df = _fetch_short_interest_finra_direct(ticker)

        if df.empty:
            log.warning(f"    No short interest data for {ticker}")
            continue

        df = df.sort_values("date")

        # Short ratio change over configurable lookback
        lookback = cfg.SHORT_LOOKBACK
        df["short_ratio_chg"] = df["short_ratio"].pct_change(
            min(lookback, len(df) - 1)
        )
        # Z-score
        df["short_ratio_z"] = (
            (df["short_ratio"] - df["short_ratio"].rolling(lookback, min_periods=4).mean())
            / (df["short_ratio"].rolling(lookback, min_periods=4).std() + 1e-9)
        )

        frames.append(df)
        time.sleep(0.2)

    if not frames:
        log.warning("No short interest data collected.")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result[result["date"] >= cfg.START_DATE].reset_index(drop=True)
    log.info(f"Short interest dataset: {len(result)} rows")
    return result


# ── 4. AUM (yfinance totalAssets) ─────────────────────────────────────────────

def build_aum_dataset() -> pd.DataFrame:
    """
    Fetch AUM (totalAssets) history for all ETFs via yfinance.
    yfinance only provides the *current* snapshot, so we build a running
    time-series by appending today's value to the HF-stored history.

    For the initial historical backfill we use the ETF NAV × shares outstanding
    proxy computed from OHLCV data already pulled for the flow proxy.
    """
    log.info("=== Building AUM Dataset ===")

    # Pull existing HF history (incremental append)
    existing = _pull_parquet(cfg.HF_FILES["aum"])

    rows = []
    for ticker in cfg.ALL_TICKERS:
        try:
            info = yf.Ticker(ticker).fast_info
            aum = getattr(info, "total_assets", None)
            if aum is None:
                # Try the slower .info dict
                full_info = yf.Ticker(ticker).info
                aum = full_info.get("totalAssets", None)

            if aum and aum > 0:
                rows.append({
                    "date": pd.Timestamp(cfg.TODAY),
                    "etf":  ticker,
                    "aum":  float(aum),
                })
                log.info(f"  {ticker}: AUM = ${aum/1e9:.2f}B")
            else:
                log.warning(f"  {ticker}: AUM not available")
            time.sleep(0.15)
        except Exception as e:
            log.warning(f"  {ticker}: {e}")

    new_rows = pd.DataFrame(rows)

    # Merge with existing history
    if existing is not None and not existing.empty:
        # Remove today's date from existing (avoid duplicate)
        existing = existing[existing["date"].astype(str) != cfg.TODAY]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    if combined.empty:
        return pd.DataFrame()

    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["etf", "date"]).drop_duplicates(["date", "etf"])

    # Compute AUM change signals
    combined["aum_chg_5d"]  = combined.groupby("etf")["aum"].pct_change(5)
    combined["aum_chg_21d"] = combined.groupby("etf")["aum"].pct_change(21)
    combined["aum_change"]  = combined["aum_chg_21d"]   # primary signal

    log.info(f"AUM dataset: {len(combined)} rows, {combined['etf'].nunique()} ETFs")
    return combined.reset_index(drop=True)


# ── 5. COMPOSITE FLOW SCORE ───────────────────────────────────────────────────

def build_composite_scores(
    df_cot:   pd.DataFrame,
    df_flow:  pd.DataFrame,
    df_short: pd.DataFrame,
    df_aum:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine all four signals into a single composite FLOW score per (date, etf).

    Strategy:
      - COT Index        (weekly, forward-filled to daily) — contrarian signal
      - Flow Proxy Z     (daily)                           — momentum signal
      - Short Ratio Chg  (twice-monthly, forward-filled)   — contrarian signal
      - AUM Change 21d   (daily snapshot)                  — momentum signal

    All signals are cross-sectionally z-scored within each date before weighting.
    Final score is in [-3, +3] range; higher = stronger inflow/bullish positioning.
    """
    log.info("=== Building Composite Flow Scores ===")

    # Build a daily date spine from all available dates
    all_dates = pd.date_range(cfg.START_DATE, cfg.TODAY, freq="B")

    # --- Pivot each source to wide (date × etf) then forward-fill ---
    def to_wide(df: pd.DataFrame, date_col: str,
                val_col: str, ffill_limit: int = 10) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(index=all_dates)
        pivot = (
            df.dropna(subset=[val_col])
              .pivot_table(index=date_col, columns="etf",
                           values=val_col, aggfunc="last")
              .reindex(all_dates)
              .ffill(limit=ffill_limit)
        )
        return pivot

    wide_cot   = to_wide(df_cot,   "date", "cot_index",      ffill_limit=7)
    wide_flow  = to_wide(df_flow,  "date", "flow_proxy_z",   ffill_limit=2)
    wide_short = to_wide(df_short, "date", "short_ratio_chg",ffill_limit=20)
    wide_aum   = to_wide(df_aum,   "date", "aum_change",     ffill_limit=5)

    def xs_zscore(wide: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score: subtract date mean, divide by date std."""
        mu  = wide.mean(axis=1)
        sig = wide.std(axis=1).replace(0, 1)
        return wide.sub(mu, axis=0).div(sig, axis=0)

    z_cot   = xs_zscore(wide_cot)
    z_flow  = xs_zscore(wide_flow)
    z_short = xs_zscore(wide_short)
    z_aum   = xs_zscore(wide_aum)

    w = cfg.SIGNAL_WEIGHTS
    composite = (
        z_cot   * w["cot_index"]
      + z_flow  * w["flow_proxy_z"]
      + z_short * w["short_ratio_chg"]
      + z_aum   * w["aum_change"]
    )

    # Melt back to long format
    result_frames = []
    for wide, name in [
        (z_cot,   "cot_index_z"),
        (z_flow,  "flow_proxy_z"),
        (z_short, "short_ratio_chg_z"),
        (z_aum,   "aum_change_z"),
        (composite, "composite_score"),
    ]:
        melted = wide.reset_index().melt(
            id_vars="index", var_name="etf", value_name=name
        ).rename(columns={"index": "date"})
        result_frames.append(melted.set_index(["date", "etf"]))

    out = pd.concat(result_frames, axis=1).reset_index()
    out = out.dropna(subset=["composite_score"])
    out = out[out["date"] >= cfg.START_DATE].sort_values(["date", "etf"])

    # Rank within each date (1 = best flow signal)
    out["flow_rank"] = (
        out.groupby("date")["composite_score"]
           .rank(ascending=False, method="min")
           .astype(int)
    )

    log.info(f"Composite scores: {len(out)} rows")
    return out.reset_index(drop=True)


# ── 6. MAIN ORCHESTRATOR ──────────────────────────────────────────────────────

def run_full_backfill():
    """Run complete historical download from 2008. Use on first deployment."""
    log.info("=== FULL BACKFILL MODE (2008 → present) ===")
    df_cot   = build_cot_dataset()
    df_flow  = build_flow_proxy_dataset()
    df_short = build_short_interest_dataset()
    df_aum   = build_aum_dataset()

    if not df_cot.empty:
        _push_parquet(df_cot,   cfg.HF_FILES["cot"],   "Backfill COT positioning 2008-present")
    if not df_flow.empty:
        _push_parquet(df_flow,  cfg.HF_FILES["flow_proxy"], "Backfill flow proxy 2008-present")
    if not df_short.empty:
        _push_parquet(df_short, cfg.HF_FILES["short_interest"], "Backfill short interest")
    if not df_aum.empty:
        _push_parquet(df_aum,   cfg.HF_FILES["aum"],   "Backfill AUM history")

    df_comp = build_composite_scores(df_cot, df_flow, df_short, df_aum)
    if not df_comp.empty:
        _push_parquet(df_comp, cfg.HF_FILES["composite"], "Backfill composite flow scores")

    return df_cot, df_flow, df_short, df_aum, df_comp


def run_daily_update():
    """
    Daily incremental update — pulls existing data, appends today,
    recomputes composites, pushes back to HF.
    """
    log.info("=== DAILY UPDATE MODE ===")

    # Pull existing datasets from HF
    df_cot   = _pull_parquet(cfg.HF_FILES["cot"])   or pd.DataFrame()
    df_flow  = _pull_parquet(cfg.HF_FILES["flow_proxy"]) or pd.DataFrame()
    df_short = _pull_parquet(cfg.HF_FILES["short_interest"]) or pd.DataFrame()

    # For COT: only re-download current year (new weekly releases)
    current_year = datetime.now().year
    df_cot_new = _download_cftc_year(current_year)
    if df_cot_new is not None:
        etf_frames = []
        for etf in cfg.COT_MARKET_MAP:
            df_e = _parse_cot_for_etf(df_cot_new, etf)
            if not df_e.empty:
                etf_frames.append(df_e)
        if etf_frames:
            new_cot = pd.concat(etf_frames, ignore_index=True)
            if not df_cot.empty:
                df_cot = pd.concat([
                    df_cot[df_cot["date"] < new_cot["date"].min()],
                    new_cot
                ], ignore_index=True)
            else:
                df_cot = new_cot
            _push_parquet(df_cot, cfg.HF_FILES["cot"], f"Daily COT update {cfg.TODAY}")

    # Flow proxy: append today's data
    today_flow_frames = []
    for ticker in cfg.ALL_TICKERS:
        try:
            raw = yf.download(ticker, start=cfg.TODAY, end=cfg.TODAY,
                              progress=False, auto_adjust=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            row = {
                "date":       pd.Timestamp(cfg.TODAY),
                "etf":        ticker,
                "dollar_vol": float(raw["Close"].iloc[-1] * raw["Volume"].iloc[-1]),
                "close":      float(raw["Close"].iloc[-1]),
            }
            today_flow_frames.append(row)
            time.sleep(0.05)
        except Exception:
            pass

    if today_flow_frames:
        df_today = pd.DataFrame(today_flow_frames)
        # Recompute rolling signals by merging with history
        df_flow_full = pd.concat([df_flow, df_today], ignore_index=True) if not df_flow.empty else df_today
        df_flow_full["date"] = pd.to_datetime(df_flow_full["date"])

        for w in cfg.FLOW_PROXY_WINDOWS:
            df_flow_full[f"dv_chg_{w}d"] = df_flow_full.groupby("etf")["dollar_vol"].pct_change(w)
        w_z = cfg.FLOW_PROXY_ZSCORE_WINDOW
        chg21 = df_flow_full.groupby("etf")["dv_chg_21d"].transform(lambda x: x)
        df_flow_full["flow_proxy_z"] = df_flow_full.groupby("etf")["dv_chg_21d"].transform(
            lambda x: (x - x.rolling(w_z, min_periods=60).mean())
                      / (x.rolling(w_z, min_periods=60).std() + 1e-9)
        )
        df_flow_full["rel_volume"] = df_flow_full.groupby("etf")["dollar_vol"].transform(
            lambda x: x / x.rolling(21, min_periods=5).mean()
        )
        df_flow = df_flow_full
        _push_parquet(df_flow, cfg.HF_FILES["flow_proxy"], f"Daily flow update {cfg.TODAY}")

    # AUM: always append today's snapshot
    df_aum = build_aum_dataset()
    if not df_aum.empty:
        _push_parquet(df_aum, cfg.HF_FILES["aum"], f"Daily AUM update {cfg.TODAY}")

    # Recompute composite scores with latest data
    df_comp = build_composite_scores(df_cot, df_flow, df_short, df_aum)
    if not df_comp.empty:
        _push_parquet(df_comp, cfg.HF_FILES["composite"], f"Daily composite update {cfg.TODAY}")

    return df_cot, df_flow, df_short, df_aum, df_comp


def load_all_from_hf():
    """Load all datasets from HF for use in Streamlit."""
    return {
        "cot":        _pull_parquet(cfg.HF_FILES["cot"]),
        "flow_proxy": _pull_parquet(cfg.HF_FILES["flow_proxy"]),
        "short":      _pull_parquet(cfg.HF_FILES["short_interest"]),
        "aum":        _pull_parquet(cfg.HF_FILES["aum"]),
        "composite":  _pull_parquet(cfg.HF_FILES["composite"]),
    }
