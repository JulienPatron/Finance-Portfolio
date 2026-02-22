import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime

# --- PAGE CONFIGURATION ---
# Note: st.set_page_config is usually handled by main.py if this is a multipage app
# st.set_page_config(page_title="Project Portfolio - Julien Patron", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    h1, h2, h3 {font-family: 'Roboto', sans-serif; font-weight: 400;}
    /* Metrics styling adjusted for Dark Mode compatibility */
    .stMetric {padding: 5px; border-radius: 5px;} 
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. SIDEBAR CONFIGURATION
# ==============================================================================
st.sidebar.header("Configuration")

# A. Assets
default_tickers = "AAPL, MSFT, TSLA, JPM, KO, WMT, V, BA, ADBE, NKE, KKR, GM"
tickers_input = st.sidebar.text_input("Stocks (comma separated)", value=default_tickers)
tickers = [x.strip().upper() for x in tickers_input.split(',') if x.strip()]

# B. Benchmark
benchmark_input = st.sidebar.text_input("Benchmark", value="SPY").strip().upper()

# C. Time Horizon
years_back = st.sidebar.slider("Historical Data (Years)", min_value=1, max_value=10, value=3)

# ==============================================================================
# 2. DATA ENGINE
# ==============================================================================
st.title("Equity Valuation Model (CAPM)")
st.markdown("Analytical tool based on the Capital Asset Pricing Model (CAPM) to identify undervalued stocks using the Security Market Line.")
st.markdown("---")

if not tickers or not benchmark_input:
    st.error("Please enter valid tickers and a benchmark.")
    st.stop()

with st.spinner('Fetching Market Data & Running Regression...'):
    # Timeframe
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=years_back*365)

    # 1. Get Risk Free Rate Data
    try:
        tnx = yf.download("^TNX", start=start_date, end=end_date, progress=False)
        
        # Handle MultiIndex
        if isinstance(tnx.columns, pd.MultiIndex):
            tnx = tnx['Close']
        elif 'Close' in tnx.columns:
            tnx = tnx['Close']
            
        # FIX TIMEZONE
        tnx.index = pd.to_datetime(tnx.index).tz_localize(None)
        
        rf_series = tnx.dropna() / 100 
        
        if not rf_series.empty:
            rf_current = rf_series.iloc[-1]
            if isinstance(rf_current, pd.Series): 
                rf_current = rf_current.iloc[0]
            rf_history_avg = rf_series.mean()
            if isinstance(rf_history_avg, pd.Series):
                rf_history_avg = rf_history_avg.iloc[0]
        else:
            rf_current = 0.04
            rf_history_avg = 0.04
            
    except Exception as e:
        rf_current = 0.04
        rf_history_avg = 0.04

    # 2. Download Stock & Benchmark Data
    all_symbols = tickers + [benchmark_input]
    try:
        raw_data = yf.download(all_symbols, start=start_date, end=end_date, progress=False)
        
        # DATA NORMALIZATION
        if isinstance(raw_data.columns, pd.MultiIndex):
            data = raw_data['Close']
        elif 'Close' in raw_data.columns:
            data = raw_data['Close']
        else:
            data = raw_data 
            
        # FIX TIMEZONE (Critical)
        data.index = pd.to_datetime(data.index).tz_localize(None)
        
        if benchmark_input not in data.columns:
            st.error(f"Benchmark '{benchmark_input}' not found. Columns: {list(data.columns)}")
            st.stop()
            
    except Exception as e:
        st.error(f"Data Download Error: {e}")
        st.stop()

    # 3. Calculate Returns
    returns = data.pct_change()
    bench_ret = returns[benchmark_input]
    stock_rets = returns.drop(columns=[benchmark_input], errors='ignore')

    # ==============================================================================
    # 3. QUANTITATIVE MODEL (CAPM REGRESSION)
    # ==============================================================================
    
    capm_data = []
    TRADING_DAYS = 252 
    rf_daily = rf_history_avg / TRADING_DAYS
    market_excess_series = bench_ret - rf_daily

    for ticker in stock_rets.columns:
        if ticker == benchmark_input: continue

        stock_excess_series = stock_rets[ticker] - rf_daily

        # PAIRWISE CLEANING
        df_reg = pd.concat([market_excess_series, stock_excess_series], axis=1)
        df_reg.columns = ['Market', 'Stock']
        df_reg = df_reg.dropna()

        if len(df_reg) < 30: continue

        x = df_reg['Market'].values
        y = df_reg['Stock'].values

        try:
            beta, alpha_daily = np.polyfit(x, y, 1)
        except:
            continue
        
        r_squared = 0
        if len(x) > 2:
            correlation_matrix = np.corrcoef(x, y)
            correlation_xy = correlation_matrix[0,1]
            r_squared = correlation_xy**2

        alpha_annual = alpha_daily * TRADING_DAYS
        mkt_annual_ret = df_reg['Market'].mean() * TRADING_DAYS + rf_history_avg
        expected_return = rf_current + beta * (mkt_annual_ret - rf_current)
        actual_return = df_reg['Stock'].mean() * TRADING_DAYS + rf_history_avg
        
        # Récupération du nom de l'entreprise
        try:
            company_name = yf.Ticker(ticker).info.get('shortName', ticker)
        except:
            company_name = ticker

        capm_data.append({
            'Ticker': ticker,
            'Company': company_name,
            'Beta': beta,
            'Alpha (%)': alpha_annual,
            'Actual Return': actual_return,
            'Expected Return (CAPM)': expected_return,
            'R-Squared': r_squared,
            'Valuation': 'Undervalued' if alpha_annual > 0 else 'Overvalued'
        })

    if not capm_data:
        st.error("❌ No valid data found. Check tickers or timeframe.")
        st.stop()

    df_capm = pd.DataFrame(capm_data).set_index('Ticker')

    # ==============================================================================
    # 4. VISUALIZATION DASHBOARD (PLOTLY INTERACTIVE)
    # ==============================================================================

    # --- SIDEBAR METRICS ---
    mkt_return_ann = bench_ret.mean() * 252
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Market Metrics")
    st.sidebar.metric("Market Return (Ann.)", f"{mkt_return_ann:.1%}")
    st.sidebar.metric("Risk-Free Rate (10Y US Bond)", f"{rf_current:.2%}")
    st.sidebar.metric("Market Risk Prem.", f"{(mkt_return_ann - rf_current):.1%}")

    # --- MAIN CONTENT ---
    col_chart, col_table = st.columns([2, 1])

    # --- CHART: SML (Plotly) ---
    with col_chart:
        st.subheader("Security Market Line (SML)")
        
        fig = go.Figure()
        
        max_beta = df_capm['Beta'].max()
        if pd.isna(max_beta) or max_beta < 0.5: max_beta = 1.5
            
        # Création de la ligne SML
        x_range = np.linspace(0, max_beta * 1.2, 100)
        y_sml = rf_current + x_range * (mkt_return_ann - rf_current)
        
        fig.add_trace(go.Scatter(
            x=x_range, 
            y=y_sml, 
            mode='lines', 
            name='SML (Fair Value)', 
            line=dict(color='gray', dash='dash', width=2)
        ))

        # Création des points pour les actions
        colors = ['#228B22' if val == 'Undervalued' else '#D90429' for val in df_capm['Valuation']]
        
        fig.add_trace(go.Scatter(
            x=df_capm['Beta'], 
            y=df_capm['Actual Return'], 
            mode='markers+text',
            text=df_capm.index,
            textposition="top center",
            showlegend=False, # <-- MODIFICATION ICI : Cache la légende pour les points
            marker=dict(size=14, color=colors, line=dict(width=1, color='white')),
            hovertemplate="<b>%{text}</b><br>Beta: %{x:.2f}<br>Return: %{y:.1%}<extra></extra>"
        ))

        fig.update_layout(
            xaxis_title='Beta (Systematic Risk)',
            yaxis_title='Annualized Return',
            yaxis_tickformat='.0%',
            xaxis=dict(range=[0, max_beta * 1.2]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

# --- TABLE: DETAILS ---
    with col_table:
        st.subheader("Alpha Generation")
        
        # Réorganisation de l'ordre : Company -> Valuation (Signal) -> Beta -> Alpha
        df_display = df_capm[['Company', 'Valuation', 'Beta', 'Alpha (%)']].copy()
        df_display = df_display.sort_values(by='Alpha (%)', ascending=False)
        
        # Formatage pour l'affichage propre
        df_display['Alpha (%)'] = df_display['Alpha (%)'].apply(lambda x: f"{x:.2%}")
        df_display['Beta'] = df_display['Beta'].apply(lambda x: f"{x:.2f}")

        st.dataframe(
            df_display,
            use_container_width=True,
            column_config={
                "Company": "Company Name",
                "Valuation": st.column_config.TextColumn(
                    "Signal",
                    help="Undervalued = Above SML",
                    width="medium"
                ),
                "Beta": "Beta",
                "Alpha (%)": "Alpha"
            }
        )