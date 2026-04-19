"""
Streamlit Dashboard for EVT Tail Risk Results.
Professional layout with dark mode support.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from huggingface_hub import HfApi, hf_hub_download
import json
import os
from datetime import datetime, timedelta
import config

st.set_page_config(
    page_title="P2Quant EVT Tail Risk Monitor",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 1rem;
    }
    .warning-badge {
        background-color: #d9534f;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_data(ttl=3600)
def load_latest_results():
    """Fetch the most recent result file from HF dataset."""
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None, None
        latest_file = json_files[0]
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO,
            filename=latest_file,
            repo_type="dataset",
            token=config.HF_TOKEN,
            cache_dir="./hf_cache"
        )
        with open(local_path, 'r') as f:
            data = json.load(f)
        return data, latest_file
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None, None

@st.cache_data(ttl=3600)
def load_historical_tail_shapes(ticker: str, lookback_days: int = 90):
    """Load tail shape history for a specific ticker from recent files."""
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)[:lookback_days]
        
        dates = []
        shapes = []
        shapes_smooth = []
        warnings = []
        
        for f in json_files:
            date_str = f.replace("evt_tailrisk_", "").replace(".json", "")
            try:
                local_path = hf_hub_download(
                    repo_id=config.HF_OUTPUT_REPO,
                    filename=f,
                    repo_type="dataset",
                    token=config.HF_TOKEN,
                    cache_dir="./hf_cache"
                )
                with open(local_path, 'r') as file:
                    data = json.load(file)
                # Search all universes for the ticker
                for universe, tickers_data in data['universes'].items():
                    if ticker in tickers_data:
                        dates.append(date_str)
                        shapes.append(tickers_data[ticker].get('tail_shape', None))
                        shapes_smooth.append(tickers_data[ticker].get('tail_shape_smooth', None))
                        warnings.append(tickers_data[ticker].get('tail_warning', 0))
                        break
            except:
                continue
                
        df = pd.DataFrame({
            'Date': pd.to_datetime(dates),
            'tail_shape': shapes,
            'tail_shape_smooth': shapes_smooth,
            'tail_warning': warnings
        }).sort_values('Date')
        
        return df
    except Exception as e:
        st.error(f"Error loading history for {ticker}: {e}")
        return pd.DataFrame()

def display_warning_badge(is_warning: bool):
    if is_warning:
        return '<span class="warning-badge">⚠️ WARNING</span>'
    return '<span style="color: green;">✅ Normal</span>'

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown(f"**Data Source:** `{config.HF_DATA_REPO}`")
st.sidebar.markdown(f"**Results Repo:** `{config.HF_OUTPUT_REPO}`")
st.sidebar.divider()

st.sidebar.markdown("### 📊 EVT Parameters")
st.sidebar.markdown(f"- Threshold Quantile: **{config.EVT_THRESHOLD_QUANTILE}**")
st.sidebar.markdown(f"- Rolling Window: **{config.ROLLING_WINDOW} days**")
st.sidebar.markdown(f"- Tail Warning Threshold (ξ): **{config.TAIL_SHAPE_WARNING_THRESHOLD}**")
st.sidebar.divider()

st.sidebar.markdown("### 🕒 Last Updated")
data, latest_file = load_latest_results()
if latest_file:
    run_date = data.get('run_date', 'Unknown')
    st.sidebar.markdown(f"**{run_date}**")
else:
    st.sidebar.markdown("*No data available*")

st.sidebar.divider()
st.sidebar.markdown("### 📖 About")
st.sidebar.markdown("""
**EVT Tail Risk Engine** estimates extreme loss probabilities using Peaks-Over-Threshold (POT) with Generalized Pareto Distribution (GPD).

- **ξ (Shape):** Tail heaviness (>0.3 triggers warning)
- **VaR 99%:** 1-day Value-at-Risk at 99% confidence
- **ES 99%:** Expected Shortfall (average loss beyond VaR)
""")

# --- Main Content ---
st.markdown('<div class="main-header">⚠️ P2Quant EVT Tail Risk Monitor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Extreme Value Theory (Peaks-Over-Threshold) Analysis</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available. Please run the daily pipeline first.")
    st.stop()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📋 Current Tail Risk Dashboard", "📈 Historical Analysis", "📊 Universe Overview"])

with tab1:
    st.markdown("### Today's Tail Risk Summary")
    
    # Universe selector
    universe_options = list(data['universes'].keys())
    selected_universe = st.selectbox("Select Universe", universe_options, index=2)  # Default to COMBINED
    
    universe_data = data['universes'][selected_universe]
    
    # Convert to DataFrame for display
    rows = []
    for ticker, metrics in universe_data.items():
        rows.append({
            'Ticker': ticker,
            'Tail Shape (ξ)': f"{metrics.get('tail_shape', 0):.3f}",
            'Smoothed ξ': f"{metrics.get('tail_shape_smooth', 0):.3f}",
            'VaR 99%': f"{metrics.get('var_99', 0)*100:.2f}%" if metrics.get('var_99') else 'N/A',
            'ES 99%': f"{metrics.get('es_99', 0)*100:.2f}%" if metrics.get('es_99') else 'N/A',
            'Warning': metrics.get('tail_warning', 0)
        })
    
    df_display = pd.DataFrame(rows)
    
    # Color-coded table
    def highlight_warning(row):
        if row['Warning'] == 1:
            return ['background-color: #ffcccc'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        df_display.style.apply(highlight_warning, axis=1),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Warning": st.column_config.CheckboxColumn("Warning Flag")
        }
    )
    
    # Metrics cards for top warnings
    warning_tickers = [t for t, m in universe_data.items() if m.get('tail_warning', 0) == 1]
    if warning_tickers:
        st.markdown("### ⚠️ Active Tail Risk Warnings")
        cols = st.columns(min(len(warning_tickers), 4))
        for i, ticker in enumerate(warning_tickers[:4]):
            with cols[i]:
                m = universe_data[ticker]
                st.metric(
                    label=f"{ticker} Tail Shape",
                    value=f"{m.get('tail_shape_smooth', 0):.3f}",
                    delta=f"Raw: {m.get('tail_shape', 0):.3f}",
                    delta_color="off"
                )
                st.caption(f"VaR 99%: {m.get('var_99', 0)*100:.2f}%")
                st.caption(f"ES 99%: {m.get('es_99', 0)*100:.2f}%")

with tab2:
    st.markdown("### Historical Tail Shape (ξ) Analysis")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_ticker = st.selectbox("Select ETF", sorted(config.ALL_TICKERS))
        lookback = st.slider("Lookback Days", 30, 365, 90)
    
    with col2:
        if selected_ticker:
            hist_df = load_historical_tail_shapes(selected_ticker, lookback)
            if not hist_df.empty:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(
                        x=hist_df['Date'], y=hist_df['tail_shape_smooth'],
                        name="Smoothed ξ", line=dict(color='blue', width=2)
                    ),
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(
                        x=hist_df['Date'], y=hist_df['tail_shape'],
                        name="Raw ξ", line=dict(color='gray', width=1, dash='dot')
                    ),
                    secondary_y=False
                )
                
                # Add warning threshold line
                fig.add_hline(
                    y=config.TAIL_SHAPE_WARNING_THRESHOLD, 
                    line_dash="dash", line_color="red",
                    annotation_text="Warning Threshold"
                )
                
                # Shade regions where warning was active
                warning_periods = hist_df[hist_df['tail_warning'] == 1]
                if not warning_periods.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=warning_periods['Date'], y=[config.TAIL_SHAPE_WARNING_THRESHOLD]*len(warning_periods),
                            mode='markers', marker=dict(color='red', size=8, symbol='x'),
                            name='Warning Active'
                        ),
                        secondary_y=False
                    )
                
                fig.update_layout(
                    title=f"{selected_ticker} Tail Shape Evolution (ξ)",
                    xaxis_title="Date",
                    yaxis_title="Tail Shape (ξ)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display stats
                st.markdown(f"**Latest Smoothed ξ:** {hist_df['tail_shape_smooth'].iloc[-1]:.3f}")
                st.markdown(f"**Warning Status:** {display_warning_badge(hist_df['tail_warning'].iloc[-1] == 1)}", unsafe_allow_html=True)
            else:
                st.info(f"No historical data available for {selected_ticker}")

with tab3:
    st.markdown("### Universe-Wide Tail Risk Heatmap")
    
    # Gather latest tail shapes for all tickers
    all_tickers_data = {}
    for universe, tickers_data in data['universes'].items():
        for ticker, metrics in tickers_data.items():
            all_tickers_data[ticker] = {
                'universe': universe,
                'tail_shape_smooth': metrics.get('tail_shape_smooth'),
                'tail_warning': metrics.get('tail_warning', 0)
            }
    
    df_all = pd.DataFrame.from_dict(all_tickers_data, orient='index')
    df_all = df_all.sort_values('tail_shape_smooth', ascending=False)
    
    # Bar chart of tail shapes
    fig = go.Figure()
    colors = ['red' if w == 1 else 'steelblue' for w in df_all['tail_warning']]
    fig.add_trace(go.Bar(
        x=df_all.index,
        y=df_all['tail_shape_smooth'],
        marker_color=colors,
        text=df_all['tail_shape_smooth'].round(3),
        textposition='outside'
    ))
    fig.add_hline(
        y=config.TAIL_SHAPE_WARNING_THRESHOLD,
        line_dash="dash", line_color="red",
        annotation_text="Warning Threshold"
    )
    fig.update_layout(
        title="Current Smoothed Tail Shape (ξ) by ETF",
        xaxis_title="ETF Ticker",
        yaxis_title="Smoothed Tail Shape (ξ)",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Risk Ranking")
    st.dataframe(
        df_all.style.background_gradient(subset=['tail_shape_smooth'], cmap='Reds'),
        use_container_width=True,
        column_config={
            "tail_warning": st.column_config.CheckboxColumn("Warning Flag")
        }
    )
