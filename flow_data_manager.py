"""
flow_data_manager.py
Core data-building and update functions shared by:
  - flow_seed.py     (one-time historical backfill)
  - flow_trainer.py  (daily incremental update, called by GitHub Actions)

Sources:
  1. COT          — CFTC disaggregated futures zips (free, weekly)
  2. Flow Proxy   — yfinance OHLCV → dollar-volume momentum (free, daily)
  3. Short Interest— FINRA direct API no key needed (~2yr history)
                     OR Nasdaq Data Link with NASDAQ_API_KEY (longer history)
  4. AUM          — yfinance totalAssets snapshot, appended daily
"""

import io
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
from huggingface_hub import HfApi, hf_hub_download

import flow_config as cfg

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HF helpers  — _pull_parquet returns None, never raises
# ─────────────────────────────────────────────────────────────────────────────

def _hf_api() -> HfApi:
    return HfApi(token=cfg.HF_TOKEN)


def _pull_parquet(hf_path: str) -> Optional[pd.DataFrame]:
    """Download parquet from HF. Returns DataFrame or None — never raises."""
    try:
        local = hf_hub_download(
            repo_id=cfg.HF_FLOW_REPO,
            filename=hf_path,
            repo_type="dataset",
            token=cfg.HF_TOKEN,
            cache_dir="./hf_cache",
            force_download=True,
        )
        df = pd.read_parquet(local)
        return df if not df.empty else None
    except Exception as e:
        log.warning(f"Could not pull {hf_path}: {e}")
        return None


def _push_parquet(df: Optional[pd.DataFrame], hf_path: str, msg: str) -> bool:
    """Push DataFrame as parquet to HF. Returns True on success."""
    if df is None or df.empty:
        log.warning(f"Skipping push of {hf_path} — empty DataFrame.")
        return False
    if not cfg.HF_TOKEN:
        log.warning("HF_TOKEN not set — skipping upload.")
        return False
    try:
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
        log.info(f"  Pushed {hf_path} ({len(df):,} rows)")
        return True
    except Exception as e:
        log.error(f"  Failed to push {hf_path}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 1. COT DATA  (CFTC disaggregated futures)
# ─────────────────────────────────────────────────────────────────────────────

CFTC_BASE = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"


def _download_cftc_year(year: int) -> Optional[pd.DataFrame]:
    url = CFTC_BASE.format(year=year)
    try:
        r = requests.get(url, timeout=90)
        if r.status_code != 200:
            log.warning(f"  CFTC {year}: HTTP {r.status_code}")
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            fname = [n for n in z.namelist() if n.endswith(".txt")][0]
            with z.open(fname) as f:
                df = pd.read_csv(f, low_memory=False)
        log.info(f"  CFTC {year}: {len(df):,} rows")
        return df
    except Exception as e:
        log.warning(f"  CFTC {year}: {e}")
        return None


def _detect_date_col(df: pd.DataFrame) -> Optional[str]:
    """Find the date column — varies across CFTC annual files."""
    for c in ["Report_Date_as_YYYY-MM-DD", "As_of_Date_In_Form_YYMMDD",
              "Report_Date_as_MM_DD_YYYY"]:
        if c in df.columns:
            return c
    # fallback
    for c in df.columns:
        if "date" in c.lower():
            return c
    return None


def _parse_date_col(series: pd.Series) -> pd.Series:
    """Parse CFTC date column that may be YYMMDD, YYYY-MM-DD, or MM/DD/YYYY."""
    s = series.astype(str).str.strip()
    for fmt in ["%Y-%m-%d", "%y%m%d", "%m/%d/%Y"]:
        result = pd.to_datetime(s, format=fmt, errors="coerce")
        if result.notna().sum() > len(s) * 0.5:
            return result
    return pd.to_datetime(s, infer_datetime_format=True, errors="coerce")


def _parse_cot_for_etf(df_all: pd.DataFrame, etf: str) -> pd.DataFrame:
    market_name = cfg.COT_MARKET_MAP.get(etf)
    if not market_name:
        return pd.DataFrame()

    col_market = cfg.COT_COLUMNS["market"]
    col_long   = cfg.COT_COLUMNS["long"]
    col_short  = cfg.COT_COLUMNS["short"]

    if col_market not in df_all.columns:
        log.warning(f"  {etf}: market column '{col_market}' not found in CFTC data")
        return pd.DataFrame()

    col_date = _detect_date_col(df_all)
    if col_date is None:
        log.warning(f"  {etf}: no date column found in CFTC data")
        return pd.DataFrame()

    keyword = market_name.upper().split(" - ")[0].strip()
    mask = df_all[col_market].astype(str).str.upper().str.contains(keyword, na=False)
    df = df_all[mask].copy()
    if df.empty:
        log.warning(f"  {etf}: keyword '{keyword}' not found in market names")
        return pd.DataFrame()

    df["date"] = _parse_date_col(df[col_date])
    df = df.dropna(subset=["date"])

    for c in [col_long, col_short]:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", "").str.strip(), errors="coerce"
            ).fillna(0)
        else:
            df[c] = 0.0

    df["net_position"] = df[col_long] - df[col_short]
    df = (df[["date", "net_position"]]
          .drop_duplicates("date")
          .sort_values("date")
          .reset_index(drop=True))
    df["etf"] = etf

    window = cfg.COT_LOOKBACK_WEEKS
    min_p  = max(4, window // 4)
    df["cot_index"] = (
        df["net_position"]
        .rolling(window, min_periods=min_p)
        .apply(
            lambda x: float((x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-9) * 100),
            raw=False,
        )
    )
    roll_mean = df["net_position"].rolling(window, min_periods=4).mean()
    roll_std  = df["net_position"].rolling(window, min_periods=4).std().fillna(1)
    df["net_position_z"] = (df["net_position"] - roll_mean) / (roll_std + 1e-9)

    return df[["date", "etf", "net_position", "net_position_z", "cot_index"]]


def build_cot_dataset() -> pd.DataFrame:
    """Full COT build from start_year to present."""
    log.info(f"=== Building COT Dataset ({start_year}–present) ===")
    frames = []
    for yr in range(2008, datetime.now().year + 1):
        df_yr = _download_cftc_year(yr)
        if df_yr is not None:
            frames.append(df_yr)
        time.sleep(0.5)

    if not frames:
        log.error("No CFTC data downloaded.")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    log.info(f"Total CFTC rows: {len(df_all):,}")

    etf_frames = []
    for etf in cfg.COT_MARKET_MAP:
        df_etf = _parse_cot_for_etf(df_all, etf)
        if not df_etf.empty:
            etf_frames.append(df_etf)
            log.info(f"  {etf}: {len(df_etf)} weekly obs")
        else:
            log.warning(f"  {etf}: no data parsed")

    if not etf_frames:
        return pd.DataFrame()

    result = pd.concat(etf_frames, ignore_index=True)
    result = result[result["date"] >= cfg.START_DATE].reset_index(drop=True)
    log.info(f"COT final: {len(result):,} rows, {result['etf'].nunique()} ETFs")
    return result


def update_cot_incremental(existing: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Re-download current year's CFTC file and merge with existing history."""
    log.info("=== COT Incremental Update ===")
    df_new_raw = _download_cftc_year(datetime.now().year)
    if df_new_raw is None:
        log.warning("Could not download current CFTC year — keeping existing data.")
        return existing if existing is not None else pd.DataFrame()

    etf_frames = []
    for etf in cfg.COT_MARKET_MAP:
        df_e = _parse_cot_for_etf(df_new_raw, etf)
        if not df_e.empty:
            etf_frames.append(df_e)

    if not etf_frames:
        log.warning("No COT data parsed from current year file.")
        return existing if existing is not None else pd.DataFrame()

    new_cot = pd.concat(etf_frames, ignore_index=True)
    new_cot["date"] = pd.to_datetime(new_cot["date"])

    if existing is not None and not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"])
        cutoff = new_cot["date"].min()
        merged = pd.concat(
            [existing[existing["date"] < cutoff], new_cot],
            ignore_index=True,
        )
    else:
        merged = new_cot

    merged = merged.sort_values(["etf", "date"]).reset_index(drop=True)
    log.info(f"COT after update: {len(merged):,} rows")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2. FLOW PROXY  (yfinance dollar-volume momentum)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_flow_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["etf", "date"]).copy()
    for w in cfg.FLOW_PROXY_WINDOWS:
        df[f"dv_chg_{w}d"] = df.groupby("etf")["dollar_vol"].pct_change(w)

    w_z = cfg.FLOW_PROXY_ZSCORE_WINDOW
    df["flow_proxy_z"] = df.groupby("etf")["dv_chg_21d"].transform(
        lambda x: (x - x.rolling(w_z, min_periods=60).mean())
                  / (x.rolling(w_z, min_periods=60).std().fillna(1) + 1e-9)
    )
    df["rel_volume"] = df.groupby("etf")["dollar_vol"].transform(
        lambda x: x / x.rolling(21, min_periods=5).mean()
    )
    return df


def build_flow_proxy_dataset() -> pd.DataFrame:
    """Full yfinance download for all ETFs, 2008–present."""
    log.info("=== Building Flow Proxy Dataset ===")
    frames = []
    for ticker in cfg.ALL_TICKERS:
        try:
            raw = yf.download(ticker, start=cfg.START_DATE, end=cfg.TODAY,
                              progress=False, auto_adjust=True)
            if raw.empty:
                log.warning(f"  {ticker}: no data")
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = pd.DataFrame({
                "date":       raw.index,
                "etf":        ticker,
                "dollar_vol": (raw["Close"] * raw["Volume"]).values,
                "close":      raw["Close"].values,
            })
            frames.append(df)
            log.info(f"  {ticker}: {len(df)} days")
            time.sleep(0.15)
        except Exception as e:
            log.warning(f"  {ticker}: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result = _compute_flow_signals(result)
    result = result.dropna(subset=["flow_proxy_z"])
    result = result[result["date"] >= cfg.START_DATE].reset_index(drop=True)
    log.info(f"Flow proxy: {len(result):,} rows")
    return result


def update_flow_proxy_incremental(existing: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Append last 5 days of OHLCV and recompute signals."""
    log.info("=== Flow Proxy Incremental Update ===")
    start = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    new_rows = []
    for ticker in cfg.ALL_TICKERS:
        try:
            raw = yf.download(ticker, start=start, end=cfg.TODAY,
                              progress=False, auto_adjust=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            for idx_val, row in raw.iterrows():
                new_rows.append({
                    "date":       pd.Timestamp(idx_val),
                    "etf":        ticker,
                    "dollar_vol": float(row["Close"] * row["Volume"]),
                    "close":      float(row["Close"]),
                })
            time.sleep(0.1)
        except Exception as e:
            log.warning(f"  {ticker}: {e}")

    if not new_rows:
        log.warning("No new flow rows — keeping existing.")
        return existing if existing is not None else pd.DataFrame()

    df_new = pd.DataFrame(new_rows)
    df_new["date"] = pd.to_datetime(df_new["date"])

    if existing is not None and not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"])
        cutoff = df_new["date"].min()
        base_cols = ["date", "etf", "dollar_vol", "close"]
        base = existing[existing["date"] < cutoff][[c for c in base_cols if c in existing.columns]]
        combined = pd.concat([base, df_new], ignore_index=True)
    else:
        combined = df_new

    combined = _compute_flow_signals(combined)
    combined = combined.dropna(subset=["flow_proxy_z"]).reset_index(drop=True)
    log.info(f"Flow proxy after update: {len(combined):,} rows")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 3. SHORT INTEREST
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_short_finra_direct(ticker: str) -> pd.DataFrame:
    """FINRA RegSho — free, no key, ~2yr history."""
    try:
        url = (
            "https://api.finra.org/data/group/OTCMarket/name/regShoDaily"
            f"?limit=5000&offset=0"
            f"&fields=issueSymbolIdentifier,shortParQuantity,totalParQuantity,settlementDate"
            f"&compareFilters=issueSymbolIdentifier:equalTo:{ticker}"
        )
        r = requests.get(url, timeout=45, headers={"Accept": "application/json"})
        if r.status_code != 200:
            return pd.DataFrame()
        rows = r.json()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"]         = pd.to_datetime(df.get("settlementDate", ""), errors="coerce")
        df["short_volume"] = pd.to_numeric(df.get("shortParQuantity", 0), errors="coerce")
        df["total_volume"] = pd.to_numeric(df.get("totalParQuantity", 1), errors="coerce")
        df["short_ratio"]  = df["short_volume"] / (df["total_volume"] + 1e-9)
        df["etf"] = ticker
        return df[["date", "etf", "short_volume", "short_ratio"]].dropna(subset=["date"])
    except Exception as e:
        log.warning(f"  {ticker} FINRA: {e}")
        return pd.DataFrame()


def _fetch_short_nasdaq(ticker: str) -> pd.DataFrame:
    """Nasdaq Data Link FINRA/SHORTS — requires NASDAQ_API_KEY."""
    if not cfg.NASDAQ_API_KEY:
        return pd.DataFrame()
    try:
        import nasdaq_data_link as ndl
        ndl.ApiConfig.api_key = cfg.NASDAQ_API_KEY
        df = ndl.get_table("FINRA/SHORTS", ticker=ticker, paginate=True)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c or "settlement" in c), None)
        vol_col  = next((c for c in df.columns if "short" in c and "vol" in c), None)
        tot_col  = next((c for c in df.columns if "total" in c and "vol" in c), None)
        if not date_col or not vol_col:
            return pd.DataFrame()
        df["date"]         = pd.to_datetime(df[date_col])
        df["short_volume"] = pd.to_numeric(df[vol_col], errors="coerce")
        df["total_volume"] = pd.to_numeric(df[tot_col], errors="coerce") if tot_col else 1.0
        df["short_ratio"]  = df["short_volume"] / (df["total_volume"] + 1e-9)
        df["etf"] = ticker
        return df[["date", "etf", "short_volume", "short_ratio"]].dropna(subset=["date"])
    except Exception as e:
        log.warning(f"  {ticker} Nasdaq DL: {e}")
        return pd.DataFrame()


def _add_short_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["etf", "date"]).copy()
    lb = cfg.SHORT_LOOKBACK
    df["short_ratio_chg"] = df.groupby("etf")["short_ratio"].pct_change(lb)
    df["short_ratio_z"]   = df.groupby("etf")["short_ratio"].transform(
        lambda x: (x - x.rolling(lb, min_periods=4).mean())
                  / (x.rolling(lb, min_periods=4).std().fillna(1) + 1e-9)
    )
    return df


def build_short_interest_dataset() -> pd.DataFrame:
    log.info("=== Building Short Interest Dataset ===")
    frames = []
    for ticker in cfg.ALL_TICKERS:
        df = _fetch_short_nasdaq(ticker) if cfg.NASDAQ_API_KEY else pd.DataFrame()
        if df.empty:
            df = _fetch_short_finra_direct(ticker)
        if df.empty:
            log.warning(f"  {ticker}: no data")
            continue
        frames.append(df)
        log.info(f"  {ticker}: {len(df)} obs")
        time.sleep(0.2)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result = result[result["date"] >= cfg.START_DATE]
    result = _add_short_signals(result).reset_index(drop=True)
    log.info(f"Short interest: {len(result):,} rows")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. AUM  (yfinance totalAssets daily snapshot)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_aum_snapshot() -> pd.DataFrame:
    """Fetch today's AUM for all tickers via yfinance."""
    rows = []
    for ticker in cfg.ALL_TICKERS:
        try:
            fi = yf.Ticker(ticker).fast_info
            aum = getattr(fi, "total_assets", None)
            if aum is None:
                aum = yf.Ticker(ticker).info.get("totalAssets", None)
            if aum and float(aum) > 0:
                rows.append({"date": pd.Timestamp(cfg.TODAY), "etf": ticker, "aum": float(aum)})
                log.info(f"  {ticker}: ${float(aum)/1e9:.2f}B")
            else:
                log.warning(f"  {ticker}: AUM not available")
            time.sleep(0.2)
        except Exception as e:
            log.warning(f"  {ticker}: {e}")
    return pd.DataFrame(rows)


def _add_aum_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["etf", "date"]).copy()
    df["aum_chg_5d"]  = df.groupby("etf")["aum"].pct_change(5)
    df["aum_chg_21d"] = df.groupby("etf")["aum"].pct_change(21)
    df["aum_change"]  = df["aum_chg_21d"]
    return df


def build_aum_dataset() -> pd.DataFrame:
    """Seed AUM dataset with today's snapshot only (history grows daily)."""
    log.info("=== Building AUM Dataset (initial snapshot) ===")
    df = _fetch_aum_snapshot()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = _add_aum_signals(df)
    log.info(f"AUM seed: {len(df)} rows")
    return df.reset_index(drop=True)


def update_aum_incremental(existing: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Append today's snapshot to existing AUM history and recompute signals.
    This is the only correct daily update path — never overwrites history.
    """
    log.info("=== AUM Incremental Update ===")
    today_df = _fetch_aum_snapshot()
    if today_df.empty:
        log.warning("No AUM snapshot — returning existing.")
        return existing if existing is not None else pd.DataFrame()

    today_df["date"] = pd.to_datetime(today_df["date"])

    if existing is not None and not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"])
        # Remove today's rows if they exist (idempotent)
        base = existing[existing["date"].dt.strftime("%Y-%m-%d") != cfg.TODAY]
        combined = pd.concat([base, today_df], ignore_index=True)
    else:
        combined = today_df

    combined = combined.drop_duplicates(["date", "etf"])
    combined = _add_aum_signals(combined).reset_index(drop=True)
    log.info(f"AUM after update: {len(combined):,} rows, "
             f"latest={combined['date'].max().strftime('%Y-%m-%d')}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 5. COMPOSITE FLOW SCORE
# ─────────────────────────────────────────────────────────────────────────────

def build_composite_scores(
    df_cot:   Optional[pd.DataFrame],
    df_flow:  Optional[pd.DataFrame],
    df_short: Optional[pd.DataFrame],
    df_aum:   Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine available signals into a composite FLOW score per (date, etf).
    Missing signals are gracefully skipped; weights are renormalised.
    """
    log.info("=== Building Composite Flow Scores ===")
    all_dates = pd.date_range(cfg.START_DATE, cfg.TODAY, freq="B")

    def _to_wide(df, date_col, val_col, ffill):
        if df is None or df.empty:
            return pd.DataFrame(index=all_dates)
        if val_col not in df.columns:
            return pd.DataFrame(index=all_dates)
        tmp = df.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col])
        return (
            tmp.dropna(subset=[val_col])
               .pivot_table(index=date_col, columns="etf",
                            values=val_col, aggfunc="last")
               .reindex(all_dates)
               .ffill(limit=ffill)
        )

    wide_cot   = _to_wide(df_cot,   "date", "cot_index",       7)
    wide_flow  = _to_wide(df_flow,  "date", "flow_proxy_z",    2)
    wide_short = _to_wide(df_short, "date", "short_ratio_chg", 20)
    wide_aum   = _to_wide(df_aum,   "date", "aum_change",      5)

    def _xs_z(wide):
        if wide.empty:
            return wide
        mu  = wide.mean(axis=1)
        sig = wide.std(axis=1).replace(0, 1).fillna(1)
        return wide.sub(mu, axis=0).div(sig, axis=0)

    z_cot   = _xs_z(wide_cot)
    z_flow  = _xs_z(wide_flow)
    z_short = _xs_z(wide_short)
    z_aum   = _xs_z(wide_aum)

    w = cfg.SIGNAL_WEIGHTS
    slots = [
        (z_cot,   "cot_index",       w["cot_index"]),
        (z_flow,  "flow_proxy_z",    w["flow_proxy_z"]),
        (z_short, "short_ratio_chg", w["short_ratio_chg"]),
        (z_aum,   "aum_change",      w["aum_change"]),
    ]
    available = [(z, wt) for z, _, wt in slots if not z.empty]
    if not available:
        log.error("No signal data for composite.")
        return pd.DataFrame()

    wt_sum  = sum(wt for _, wt in available)
    composite = sum(z * (wt / wt_sum) for z, wt in available)

    named = [
        (z_cot,   "cot_index_z"),
        (z_flow,  "flow_proxy_z"),
        (z_short, "short_ratio_chg_z"),
        (z_aum,   "aum_change_z"),
        (composite, "composite_score"),
    ]
    parts = []
    for wide, name in named:
        if wide.empty:
            continue
        melted = (
            wide.reset_index()
                .melt(id_vars="index", var_name="etf", value_name=name)
                .rename(columns={"index": "date"})
                .set_index(["date", "etf"])
        )
        parts.append(melted)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, axis=1).reset_index()
    out = out[out["date"] >= cfg.START_DATE]
    if "composite_score" not in out.columns:
        return pd.DataFrame()

    out = out.dropna(subset=["composite_score"]).sort_values(["date", "etf"])
    out["flow_rank"] = (
        out.groupby("date")["composite_score"]
           .rank(ascending=False, method="min")
           .astype(int)
    )
    log.info(f"Composite: {len(out):,} rows, "
             f"latest={out['date'].max().strftime('%Y-%m-%d') if not out.empty else 'N/A'}")
    return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public convenience: push all datasets
# ─────────────────────────────────────────────────────────────────────────────

def push_all(df_cot, df_flow, df_short, df_aum, df_comp, prefix=""):
    _push_parquet(df_cot,   cfg.HF_FILES["cot"],            f"{prefix}COT positioning")
    _push_parquet(df_flow,  cfg.HF_FILES["flow_proxy"],     f"{prefix}Flow proxy")
    _push_parquet(df_short, cfg.HF_FILES["short_interest"], f"{prefix}Short interest")
    _push_parquet(df_aum,   cfg.HF_FILES["aum"],            f"{prefix}AUM history")
    _push_parquet(df_comp,  cfg.HF_FILES["composite"],      f"{prefix}Composite scores")
