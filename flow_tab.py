"""
flow_tab.py
Streamlit tab for the Cross-Asset Flow & Positioning module.
Imported by streamlit_app.py and rendered as a new tab alongside the EVT tabs.

Sections:
  A) Live Flow Signal Dashboard  — composite scores, top picks
  B) COT Positioning             — speculator net positioning per ETF
  C) Flow Proxy                  — dollar-volume momentum heatmap
  D) Short Interest              — short ratio trends
  E) AUM Flows                   — fund inflow/outflow
  F) Signal Deep-Dive            — individual ETF all-signal view
"""

import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import streamlit as st

import flow_config as cfg
import flow_data_manager as dm   # re-uses HF helpers

log = logging.getLogger(__name__)

# ── Cache helpers ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _load_parquet_cached(hf_path: str) -> pd.DataFrame:
    try:
        local = hf_hub_download(
            repo_id=cfg.HF_FLOW_REPO,
            filename=hf_path,
            repo_type="dataset",
            token=cfg.HF_TOKEN,
            cache_dir="./hf_cache",
            force_download=False,
        )
        return pd.read_parquet(local)
    except Exception as e:
        log.warning(f"Could not load {hf_path}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def _load_latest_json() -> dict:
    try:
        local = hf_hub_download(
            repo_id=cfg.HF_FLOW_REPO,
            filename="daily_results/flow_positioning_latest.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN,
            cache_dir="./hf_cache",
            force_download=True,
        )
        with open(local) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Could not load latest flow JSON: {e}")
        return {}


def _metric_color(val: float, signal: str) -> str:
    """Return green/red/grey based on signal semantics."""
    if pd.isna(val):
        return "off"
    positive_good = signal in ("flow_proxy_z", "aum_change_z", "composite_score")
    if positive_good:
        return "normal" if val > 0 else "inverse"
    else:  # COT contrarian: high = crowded long = caution
        return "inverse" if val > 1 else "normal"


# ── Section helpers ───────────────────────────────────────────────────────────

def _render_composite_dashboard(payload: dict):
    """Top section: composite scores table + top picks."""
    st.markdown("### 🏆 Composite Flow Signal — Current Rankings")

    universe_options = list(cfg.UNIVERSES.keys())
    sel_universe = st.selectbox(
        "Universe", universe_options,
        key="flow_universe_selector",
        index=universe_options.index("COMBINED") if "COMBINED" in universe_options else 0,
    )

    with st.expander("📘 How to interpret the Rankings", expanded=False):
        st.markdown("""
**The Composite Score** is a weighted z-score combining all four signals (COT, Flow Proxy,
Short Interest, AUM), cross-sectionally normalised within the universe on each date —
so scores are *relative rankings* within the peer group, not absolute values.

| Score | Indicator | Meaning |
|---|---|---|
| **> +0.5** | 🟢 | Above-average flow momentum — institutions positioning **into** this ETF |
| **−0.5 to +0.5** | 🟡 | Neutral — no strong positioning signal |
| **< −0.5** | 🔴 | Below-average flow — money leaving or short interest building |

**Individual z-score columns** show each signal's contribution:
- **COT Z** — speculator futures positioning relative to 52-week history
- **Flow Proxy Z** — dollar-volume momentum relative to 1-year average
- **Short Ratio Chg Z** — change in short interest vs 3-month history
- **AUM Change Z** — fund inflow/redemption momentum

**Important:** This is a *positioning* signal, not a return forecast. It tells you where
institutional money is flowing — use it as a filter or overlay on top of your EVT Tail Risk
and Siamese Ranker signals for highest-conviction trade ideas.

**Best confluence setup 🎯:** Top composite rank + EVT tail warning **not** active +
Siamese Ranker top pick = all three engines aligned.
        """)

    universe_data = payload.get("universes", {}).get(sel_universe, {})
    if not universe_data:
        st.info("No composite data available yet. Run flow_trainer.py first.")
        return

    # Top picks
    tickers_sorted = list(universe_data.keys())
    if tickers_sorted:
        col1, col2, col3 = st.columns(3)
        for i, (col, label) in enumerate([(col1, "🥇 Top Pick"), (col2, "🥈 2nd"), (col3, "🥉 3rd")]):
            if i < len(tickers_sorted):
                t = tickers_sorted[i]
                d = universe_data[t]
                with col:
                    st.metric(
                        label=f"{label}",
                        value=t,
                        delta=f"Score: {d['composite_score']:+.3f}",
                        delta_color=_metric_color(d["composite_score"], "composite_score"),
                    )

    # Full rankings table
    rows = []
    for ticker, d in universe_data.items():
        rows.append({
            "ETF":              ticker,
            "Composite Score":  round(d.get("composite_score",    0), 3),
            "COT Z":            round(d.get("cot_index_z",        0), 3),
            "Flow Proxy Z":     round(d.get("flow_proxy_z",       0), 3),
            "Short Ratio Chg Z":round(d.get("short_ratio_chg_z",  0), 3),
            "AUM Change Z":     round(d.get("aum_change_z",       0), 3),
            "Rank":             d.get("flow_rank", "-"),
        })

    df = pd.DataFrame(rows).sort_values("Composite Score", ascending=False)

    def _score_indicator(val):
        try:
            v = float(val)
            if v > 0.5:  return "🟢"
            if v < -0.5: return "🔴"
            return "🟡"
        except (TypeError, ValueError):
            return "⚪"

    df["Signal"] = df["Composite Score"].apply(_score_indicator)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Signal": st.column_config.TextColumn("", width="small"),
            "Composite Score": st.column_config.NumberColumn(format="%.3f"),
            "COT Z": st.column_config.NumberColumn(format="%.3f"),
            "Flow Proxy Z": st.column_config.NumberColumn(format="%.3f"),
            "Short Ratio Chg Z": st.column_config.NumberColumn(format="%.3f"),
            "AUM Change Z": st.column_config.NumberColumn(format="%.3f"),
        }
    )

    signal_date = payload.get("signal_date", "—")
    st.caption(f"Signal date: **{signal_date}** | Weights: "
               f"COT {cfg.SIGNAL_WEIGHTS['cot_index']:.0%} · "
               f"Flow Proxy {cfg.SIGNAL_WEIGHTS['flow_proxy_z']:.0%} · "
               f"Short {cfg.SIGNAL_WEIGHTS['short_ratio_chg']:.0%} · "
               f"AUM {cfg.SIGNAL_WEIGHTS['aum_change']:.0%}")


def _render_cot_section(df_cot: pd.DataFrame):
    """COT net positioning charts."""
    st.markdown("### 📊 COT Speculator Positioning")

    with st.expander("📘 How to interpret COT Positioning", expanded=False):
        st.markdown("""
The CFTC publishes every Friday how many futures contracts **leveraged speculators**
(hedge funds, CTAs) hold long vs short across major markets. This gives direct visibility
into institutional positioning *before* it shows up in ETF prices.

#### COT Index (0–100 percentile)
The current net long position ranked within its 52-week history:

| Reading | Positioning | Signal |
|---|---|---|
| **> 80** | Speculators very crowded **LONG** | ⚠️ Contrarian **bearish** — crowded longs tend to reverse |
| **40 – 60** | Neutral | No signal |
| **< 20** | Speculators very crowded **SHORT** | ✅ Contrarian **bullish** — short squeeze risk is high |

**Example:** GLD COT Index at 90 → hedge funds maximally long gold futures.
Historically this precedes a pullback as the crowded trade unwinds.
TLT COT Index below 20 → funds maximally short Treasuries → classic contrarian long setup.

#### Net Position Z-score
Measures how many standard deviations current positioning is from its 52-week mean.
Extreme readings (> ±2) historically precede sharp reversals.

#### ETF → Futures mapping used
| ETF | Futures proxy |
|---|---|
| GLD, GDX | COMEX Gold |
| SLV, XME | COMEX Silver |
| TLT | 30-Year Treasury (CBOT) |
| SPY | E-Mini S&P 500 |
| QQQ | Nasdaq-100 |
| XLE | WTI Crude Oil (NYMEX) |

*COT is a weekly signal — data is forward-filled to daily for the composite score.*
        """)

    if df_cot is None or df_cot.empty:
        st.warning("COT data not yet available. Run flow_trainer.py --backfill first.")
        return

    df_cot["date"] = pd.to_datetime(df_cot["date"])
    available_etfs = sorted(df_cot["etf"].unique())

    col1, col2 = st.columns([1, 3])
    with col1:
        sel_etf_cot = st.selectbox("ETF", available_etfs, key="cot_etf")
        date_range  = st.slider(
            "History (days)", 90, 1500, 365, key="cot_daterange"
        )

    sub = df_cot[df_cot["etf"] == sel_etf_cot].tail(date_range)

    with col2:
        if sub.empty:
            st.info(f"No COT data for {sel_etf_cot}")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["cot_index"],
                name="COT Index (%ile)", line=dict(color="#1f77b4", width=2),
                fill="tozeroy", fillcolor="rgba(31,119,180,0.1)"
            ))
            fig.add_hline(y=80, line_dash="dash", line_color="red",
                          annotation_text="Crowded Long (80)")
            fig.add_hline(y=20, line_dash="dash", line_color="green",
                          annotation_text="Crowded Short (20)")
            fig.add_hline(y=50, line_dash="dot", line_color="grey")
            fig.update_layout(
                title=f"{sel_etf_cot} — COT Index (Speculator Positioning Percentile)",
                yaxis_title="Percentile (0–100)",
                xaxis_title="Date",
                height=400,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Latest reading
            latest = sub.iloc[-1]
            sentiment = ("🔴 Crowded Long — Contrarian Bearish"
                         if latest["cot_index"] > 80
                         else "🟢 Crowded Short — Contrarian Bullish"
                         if latest["cot_index"] < 20
                         else "⚪ Neutral Positioning")
            st.markdown(
                f"**Latest COT Index:** {latest['cot_index']:.1f}  |  "
                f"**Net Position Z:** {latest['net_position_z']:.2f}  |  "
                f"**Signal:** {sentiment}"
            )

    # Cross-ETF COT snapshot bar chart
    st.markdown("#### Cross-ETF COT Snapshot (latest)")
    latest_cot = (
        df_cot.sort_values("date")
              .groupby("etf")
              .last()
              .reset_index()[["etf", "cot_index", "net_position_z"]]
    )
    if not latest_cot.empty:
        fig2 = px.bar(
            latest_cot.sort_values("cot_index", ascending=True),
            x="cot_index", y="etf",
            orientation="h",
            color="cot_index",
            color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
            range_color=[0, 100],
            labels={"cot_index": "COT Index (%ile)", "etf": "ETF"},
            title="Latest COT Index by ETF (>80 = crowded long warning)",
        )
        fig2.add_vline(x=80, line_dash="dash", line_color="red")
        fig2.add_vline(x=20, line_dash="dash", line_color="green")
        fig2.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)


def _render_flow_proxy_section(df_flow: pd.DataFrame):
    """Dollar-volume momentum charts."""
    st.markdown("### 💰 Flow Proxy — Dollar-Volume Momentum")

    with st.expander("📘 How to interpret Flow Proxy", expanded=False):
        st.markdown("""
**Dollar Volume = Close Price × Volume.** When institutions buy or sell an ETF in size,
dollar volume spikes. This tab measures whether current dollar volume is unusually high
or low relative to the trailing 1-year history.

#### Flow Proxy Z-score
Z-score of the 21-day dollar-volume change vs trailing 252-day history:

| Z-score | Meaning |
|---|---|
| **> +1.5** | Unusual activity — likely institutional accumulation or rebalancing |
| **0 to +1.5** | Mild positive momentum |
| **−1.5 to 0** | Mild outflow or quiet period |
| **< −1.5** | Volume has collapsed — distribution or institutional exit |

#### The Heatmap (most useful view)
Shows all ETFs × last 30 trading days at once:
- **Sustained green column** (2+ weeks) → genuine accumulation trend, not a one-day spike
- **Sustained red column** → persistent distribution — avoid or short
- **Single-day spike** → likely news event, not a positioning signal

#### Key distinction vs AUM
- **Flow Proxy** measures *trading activity* — volume spikes
- **AUM** measures *net capital* — whether money is actually staying in or leaving

Both positive together = strong institutional accumulation signal.
Flow Proxy positive + AUM falling = churning / distribution masquerading as activity.

*Dollar volume = price × volume, so a big price move on normal volume looks the same as
normal price on big volume. Best used alongside COT for confirmation.*
        """)

    if df_flow is None or df_flow.empty:
        st.warning("Flow proxy data not yet available.")
        return

    df_flow["date"] = pd.to_datetime(df_flow["date"])
    available_etfs  = sorted(df_flow["etf"].unique())

    col1, col2 = st.columns([1, 3])
    with col1:
        sel_etf_flow = st.selectbox("ETF", available_etfs, key="flow_etf")
        lookback     = st.slider("History (days)", 60, 500, 252, key="flow_lookback")

    sub = df_flow[df_flow["etf"] == sel_etf_flow].tail(lookback)

    with col2:
        if sub.empty:
            st.info(f"No flow data for {sel_etf_flow}")
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=sub["date"], y=sub["flow_proxy_z"],
                name="Flow Proxy Z-score",
                marker_color=np.where(sub["flow_proxy_z"] > 0, "#2ecc71", "#e74c3c"),
            ))
            fig.add_hline(y=1.5,  line_dash="dash", line_color="green",
                          annotation_text="Strong Inflow")
            fig.add_hline(y=-1.5, line_dash="dash", line_color="red",
                          annotation_text="Strong Outflow")
            fig.update_layout(
                title=f"{sel_etf_flow} — Dollar-Volume Flow Z-score",
                yaxis_title="Z-score",
                xaxis_title="Date",
                height=400,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Heatmap: all ETFs × last 30 days
    st.markdown("#### Flow Proxy Heatmap — All ETFs (last 30 trading days)")
    recent = df_flow[df_flow["date"] >= (pd.Timestamp(cfg.TODAY) - timedelta(days=45))]
    if not recent.empty:
        pivot = recent.pivot_table(
            index="etf", columns="date", values="flow_proxy_z", aggfunc="last"
        )
        pivot.columns = [str(c)[:10] for c in pivot.columns]
        # Keep last 30 columns
        pivot = pivot.iloc[:, -30:]
        fig_hm = px.imshow(
            pivot,
            color_continuous_scale="RdYlGn",
            zmin=-3, zmax=3,
            aspect="auto",
            title="Flow Proxy Z-score Heatmap (green = inflow, red = outflow)",
            labels={"color": "Z-score"},
        )
        fig_hm.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig_hm, use_container_width=True)


def _render_short_interest_section(df_short: pd.DataFrame):
    """Short interest ratio charts."""
    st.markdown("### 📉 Short Interest")

    with st.expander("📘 How to interpret Short Interest", expanded=False):
        st.markdown("""
**Short Ratio = Short Volume / Total Volume**, reported twice monthly by FINRA.
A rising short ratio means more participants are actively betting against the ETF.

#### Two ways to interpret a rising short ratio

**Bearish read (trend-following):**
Rising short ratio + falling price = shorts are right and momentum is down. Avoid or underweight.

**Bullish read (contrarian — the short squeeze 🚀):**
Rising short ratio + rising price = shorts are trapped and forced to cover, which
accelerates the upward move. Mining ETFs (GDX, XME) are particularly prone to violent squeezes.

#### 3-Month Short Ratio Change (primary signal)

| Change | Interpretation |
|---|---|
| **> +10%** | Shorts building significantly — warning signal or future squeeze setup |
| **−10% to +10%** | Stable positioning |
| **< −10%** | Short covering underway — bearish conviction fading, often precedes rallies |

#### Best squeeze setup
High short ratio (>40%) + rising price + COT Index turning up from below 20
= shorts are both covering *and* speculators are going long. Most powerful combination.

*Data frequency: twice monthly from FINRA. Forward-filled to daily for composite score.*
        """)

    if df_short is None or df_short.empty:
        st.info(
            "**Short interest data not yet seeded.**\n\n"
            "The seeding pipeline fetches this from FINRA (free, ~2yr history) "
            "or Nasdaq Data Link (longer history if `NASDAQ_API_KEY` is set).\n\n"
            "**Note:** `NASDAQ_API_KEY` must be added to **Streamlit Cloud secrets** "
            "(not just GitHub Actions secrets). Go to: App → Settings → Secrets and add:\n"
            "```\nNASDAQ_API_KEY = \"your_key_here\"\n```"
        )
        return

    df_short["date"] = pd.to_datetime(df_short["date"])
    available_etfs   = sorted(df_short["etf"].unique())

    col1, col2 = st.columns([1, 3])
    with col1:
        sel_etf_short = st.selectbox("ETF", available_etfs, key="short_etf")
        lookback_s    = st.slider("History (days)", 60, 730, 252, key="short_lookback")

    sub = df_short[df_short["etf"] == sel_etf_short].tail(lookback_s)

    with col2:
        if sub.empty:
            st.info(f"No short interest data for {sel_etf_short}")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["short_ratio"] * 100,
                name="Short Ratio (%)",
                line=dict(color="#9b59b6", width=2),
                fill="tozeroy", fillcolor="rgba(155,89,182,0.1)",
            ))
            fig.update_layout(
                title=f"{sel_etf_short} — Short Ratio (% of volume shorted)",
                yaxis_title="Short Ratio (%)",
                xaxis_title="Date",
                height=380,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            if "short_ratio_chg" in sub.columns and sub["short_ratio_chg"].notna().any():
                latest_chg = sub["short_ratio_chg"].iloc[-1] * 100
                direction  = "⬆️ Rising" if latest_chg > 0 else "⬇️ Falling"
                signal     = ("🔴 Shorts building" if latest_chg > 10
                              else "🟢 Short covering" if latest_chg < -10
                              else "⚪ Stable")
                st.markdown(
                    f"**3-month short ratio change:** {latest_chg:+.1f}%  |  "
                    f"**Trend:** {direction}  |  **Signal:** {signal}"
                )


def _render_aum_section(df_aum: pd.DataFrame):
    """AUM flow charts."""
    st.markdown("### 🏦 AUM Flow (Assets Under Management)")

    with st.expander("📘 How to interpret AUM Flows", expanded=False):
        st.markdown("""
**AUM (Assets Under Management)** reflects total capital in the ETF. Daily changes
combine two effects: market returns on existing holdings + net new flows (inflows minus redemptions).

#### 21-day AUM Change % (primary signal)

| Change | Meaning |
|---|---|
| **Strongly positive** | New institutional capital actively flowing in — not just price appreciation |
| **Near zero** | Stable base, price moves driven by existing holders |
| **Strongly negative** | Redemptions outpacing inflows — institutional selling pressure |

#### Reading AUM by ETF size
- **Large ETFs (SPY, QQQ, GLD)** — move slowly; need large absolute changes to be significant.
  A 1% AUM change in SPY represents ~$4B of real capital flow.
- **Small ETFs (SLV, XME, XSD, XBI)** — dramatic % swings on smaller absolute amounts.
  These are more actionable signals due to higher sensitivity.

#### Key distinction vs Flow Proxy
| Signal | Measures | Best for |
|---|---|---|
| **Flow Proxy** | Trading activity (volume spikes) | Detecting institutional *activity* |
| **AUM Change** | Net capital retained in fund | Detecting whether money *stays* or *leaves* |

**Strongest signal:** Both positive together = genuine accumulation (new money + active buying).
**Warning signal:** High Flow Proxy Z but falling AUM = churning/distribution — big volume but
money is actually leaving the fund.

*Source: yfinance totalAssets daily snapshot, accumulated into history by the daily pipeline.*
        """)

    if df_aum is None or df_aum.empty:
        st.warning("AUM data not yet available.")
        return

    df_aum["date"] = pd.to_datetime(df_aum["date"])
    available_etfs  = sorted(df_aum["etf"].unique())

    col1, col2 = st.columns([1, 3])
    with col1:
        sel_etf_aum = st.selectbox("ETF", available_etfs, key="aum_etf")
        lookback_a  = st.slider("History (days)", 30, 500, 120, key="aum_lookback")

    sub = df_aum[df_aum["etf"] == sel_etf_aum].tail(lookback_a)

    with col2:
        if sub.empty or "aum" not in sub.columns:
            st.info(f"No AUM data for {sel_etf_aum}")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["aum"] / 1e9,
                name="AUM ($B)", line=dict(color="#e67e22", width=2),
                fill="tozeroy", fillcolor="rgba(230,126,34,0.1)",
            ))
            fig.update_layout(
                title=f"{sel_etf_aum} — AUM ($B)",
                yaxis_title="AUM (Billions USD)",
                height=350, hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            if "aum_chg_21d" in sub.columns and sub["aum_chg_21d"].notna().any():
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=sub["date"], y=sub["aum_chg_21d"] * 100,
                    name="21-day AUM change (%)",
                    marker_color=np.where(sub["aum_chg_21d"] > 0, "#27ae60", "#c0392b"),
                ))
                fig2.update_layout(
                    title=f"{sel_etf_aum} — 21-day AUM Change (%)",
                    yaxis_title="%", height=280, hovermode="x unified",
                )
                st.plotly_chart(fig2, use_container_width=True)

    # Cross-ETF latest AUM snapshot
    st.markdown("#### Latest AUM by ETF")
    latest_aum = (
        df_aum.sort_values("date")
              .groupby("etf")
              .last()
              .reset_index()[["etf", "aum", "aum_chg_21d"]]
              .dropna(subset=["aum"])
              .sort_values("aum", ascending=False)
    )
    if not latest_aum.empty:
        latest_aum["AUM ($B)"] = (latest_aum["aum"] / 1e9).round(2)
        latest_aum["21d Change"] = (latest_aum["aum_chg_21d"] * 100).round(2)
        st.dataframe(
            latest_aum[["etf", "AUM ($B)", "21d Change"]].rename(columns={"etf": "ETF"}),
            use_container_width=True,
            hide_index=True,
        )


def _render_deep_dive(df_comp: pd.DataFrame, df_cot: pd.DataFrame,
                      df_flow: pd.DataFrame, df_short: pd.DataFrame,
                      df_aum: pd.DataFrame):
    """Single-ETF deep-dive showing all four signals on one chart."""
    st.markdown("### 🔍 Single-ETF Signal Deep-Dive")

    with st.expander("📘 How to use the Signal Deep-Dive", expanded=False):
        st.markdown("""
All four signals stacked on one chart for a single ETF, sharing the same time axis.
Use this to **confirm or question a top composite ranking** before acting on it.

#### What to look for: Signal Confluence

The strongest setups occur when multiple signals align simultaneously:

| Signal | Bullish Setup | Bearish Setup |
|---|---|---|
| **COT Index** | Turning up from below 20 (shorts squeezed) | Turning down from above 80 (longs crowded) |
| **Flow Proxy Z** | Positive and rising (volume momentum building) | Negative and falling (outflow) |
| **Short Ratio** | Falling (shorts covering) | Rising sharply |
| **AUM Change** | Positive (capital flowing in) | Negative (redemptions) |

**High conviction:** 3 or 4 signals aligned = trade with full size.
**Low conviction:** Only 1–2 signals positive = wait for confirmation or reduce size.

#### The highest-conviction setup (commodities ETFs: GLD, SLV, GDX)
1. COT Index at a low extreme (< 20) — speculators maximally short
2. Flow Proxy Z-score starting to turn positive — volume picking up
3. AUM change turning positive — new capital entering

This combination historically precedes sustained multi-week moves in commodity ETFs.

#### Why signals sometimes disagree
- **Flow Proxy up + COT down:** Trading activity is high but futures speculators are
  still bearish — could be retail buying into institutional distribution. Caution.
- **COT up + AUM flat:** Futures positioning is bullish but ETF investors haven't
  followed yet — early signal, watch for AUM to confirm.
- **All signals neutral:** No edge — sit on the sidelines for this ETF.
        """)

    all_etfs = sorted(cfg.ALL_TICKERS)
    sel_etf  = st.selectbox("Select ETF", all_etfs, key="deepdive_etf")
    lookback = st.slider("History (days)", 60, 730, 252, key="deepdive_lookback")

    # Build merged daily signal frame for this ETF
    frames = {}

    if df_comp is not None and not df_comp.empty:
        df_comp["date"] = pd.to_datetime(df_comp["date"])
        sub = df_comp[df_comp["etf"] == sel_etf][["date", "composite_score",
                                                    "flow_proxy_z"]].tail(lookback)
        frames["composite"] = sub.set_index("date")

    if df_cot is not None and not df_cot.empty:
        df_cot["date"] = pd.to_datetime(df_cot["date"])
        sub = df_cot[df_cot["etf"] == sel_etf][["date", "cot_index"]].tail(lookback)
        frames["cot"] = sub.set_index("date")

    if df_short is not None and not df_short.empty:
        df_short["date"] = pd.to_datetime(df_short["date"])
        sub = df_short[df_short["etf"] == sel_etf][["date", "short_ratio"]].tail(lookback)
        frames["short"] = sub.set_index("date")

    if df_aum is not None and not df_aum.empty:
        df_aum["date"] = pd.to_datetime(df_aum["date"])
        sub = df_aum[df_aum["etf"] == sel_etf][["date", "aum_chg_21d"]].tail(lookback)
        frames["aum"] = sub.set_index("date")

    if not frames:
        st.info("No signal data available yet.")
        return

    merged = pd.concat(frames.values(), axis=1).reset_index()
    if "index" in merged.columns and "date" not in merged.columns:
        merged = merged.rename(columns={"index": "date"})

    from plotly.subplots import make_subplots
    n_plots = len(frames)
    fig = make_subplots(
        rows=n_plots, cols=1, shared_xaxes=True,
        subplot_titles=[
            "Composite Flow Score",
            "COT Index (%ile)",
            "Short Ratio (%)",
            "AUM 21d Change (%)",
        ][:n_plots],
        vertical_spacing=0.06,
    )

    plot_configs = [
        ("composite_score", "#2c3e50", "Composite Score"),
        ("cot_index",       "#1f77b4", "COT Index"),
        ("short_ratio",     "#9b59b6", "Short Ratio (×100)"),
        ("aum_chg_21d",     "#e67e22", "AUM 21d Chg (×100)"),
    ]
    row = 1
    for col_name, color, label in plot_configs:
        if col_name not in merged.columns:
            continue
        y = merged[col_name] * (100 if col_name in ("short_ratio", "aum_chg_21d") else 1)
        fig.add_trace(
            go.Scatter(x=merged["date"], y=y, name=label,
                       line=dict(color=color, width=1.8)),
            row=row, col=1,
        )
        row += 1

    fig.update_layout(
        title=f"{sel_etf} — All Flow & Positioning Signals",
        height=160 * n_plots + 80,
        hovermode="x unified",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Main entry point called by streamlit_app.py ───────────────────────────────

def render_flow_tab():
    """
    Main entry point. Call this inside a `with tab:` block in streamlit_app.py.
    """
    st.markdown(
        '<h2 style="color:#1f77b4;">💧 Cross-Asset Flow & Positioning</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Four complementary positioning signals: "
        "**COT** (speculator futures positioning) · "
        "**Flow Proxy** (dollar-volume momentum) · "
        "**Short Interest** (FINRA) · "
        "**AUM Flows** (fund inflows/redemptions)",
    )

    # ── Load all data ─────────────────────────────────────────────────────────
    with st.spinner("Loading flow & positioning data from Hugging Face..."):
        payload   = _load_latest_json()
        df_cot    = _load_parquet_cached(cfg.HF_FILES["cot"])
        df_flow   = _load_parquet_cached(cfg.HF_FILES["flow_proxy"])
        df_short  = _load_parquet_cached(cfg.HF_FILES["short_interest"])
        df_aum    = _load_parquet_cached(cfg.HF_FILES["aum"])
        df_comp   = _load_parquet_cached(cfg.HF_FILES["composite"])

    if not payload:
        st.warning(
            "⚠️ No flow data found in "
            f"`{cfg.HF_FLOW_REPO}`. "
            "Run `python flow_trainer.py --backfill` first to populate the dataset."
        )
        st.code("python flow_trainer.py --backfill", language="bash")
        return

    run_date = payload.get("run_date", "—")
    st.caption(f"Last updated: **{run_date}**  |  Source: `{cfg.HF_FLOW_REPO}`")

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    sub_a, sub_b, sub_c, sub_d, sub_e, sub_f = st.tabs([
        "🏆 Rankings",
        "📊 COT Positioning",
        "💰 Flow Proxy",
        "📉 Short Interest",
        "🏦 AUM Flows",
        "🔍 Signal Deep-Dive",
    ])

    with sub_a:
        _render_composite_dashboard(payload)

    with sub_b:
        _render_cot_section(df_cot)

    with sub_c:
        _render_flow_proxy_section(df_flow)

    with sub_d:
        _render_short_interest_section(df_short)

    with sub_e:
        _render_aum_section(df_aum)

    with sub_f:
        _render_deep_dive(df_comp, df_cot, df_flow, df_short, df_aum)
