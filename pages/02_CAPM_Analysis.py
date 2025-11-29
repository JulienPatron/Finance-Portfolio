import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CAPM Alpha Hunter",
    layout="wide"
)
sns.set_theme(style="white", context="paper", font_scale=1.1)

# --- CSS STYLING (Bloomberg Style) ---
st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    h1, h2, h3 {font-family: 'Roboto', sans-serif; font-weight: 400;}
    .stMetric {background-color: #f9f9f9; padding: 10px; border-radius: 5px; border: 1px solid #e0e0e0;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. SIDEBAR CONFIGURATION
# ==============================================================================
st.sidebar.header("Configuration")

# A. Assets
default_tickers = "AAPL, MSFT, TSLA, NVDA, JPM, KO, XOM"
tickers_input = st.sidebar.text_input("Stocks (comma separated)", value=default_tickers)
tickers = [x.strip().upper() for x in tickers_input.split(',') if x.strip()]

# B. Benchmark
benchmark_input = st.sidebar.text_input("Benchmark", value="SPY").strip().upper()

# C. Time Horizon
years_back = st.sidebar.slider("Analysis Period (Years)", min_value=1, max_value=10, value=3)

# ==============================================================================
# 2. DATA ENGINE
# ==============================================================================
st.title("CAPM Alpha Hunter")
st.markdown("Capital Asset Pricing Model Analysis: Identify undervalued securities (Jensen's Alpha).")
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
        # Fetching TNX
        tnx = yf.download("^TNX", start=start_date, end=end_date, progress=False)['Close']
        
        # Clean RF Data separately
        rf_series = tnx.dropna() / 100 
        
        if not rf_series.empty:
            rf_current = rf_series.iloc[-1]
            rf_history_avg = rf_series.mean()
        else:
            raise ValueError("Empty RF Data")
            
    except Exception:
        rf_current = 0.04
        rf_history_avg = 0.04
        # On ne bloque pas l'app pour le taux sans risque, on prend une valeur par défaut
        # st.warning("Using default Risk-Free Rate (4.0%) - Connection to ^TNX failed.")

    # 2. Download Stock & Benchmark Data
    all_symbols = tickers + [benchmark_input]
    try:
        # IMPORTANT: On ne fait PAS de .dropna() global ici pour ne pas perdre les stocks
        # qui ont un historique un peu plus court ou des jours fériés différents.
        data = yf.download(all_symbols, start=start_date, end=end_date, progress=False)['Close']
        
        if benchmark_input not in data.columns:
            st.error(f"Benchmark '{benchmark_input}' not found in data.")
            st.stop()
            
    except Exception as e:
        st.error(f"Data Download Error: {e}")
        st.stop()

    # 3. Calculate Returns (Arithmetic)
    # On laisse les NaN ici, on les gérera paire par paire
    returns = data.pct_change()
    
    # Separate Benchmark and Stocks
    bench_ret = returns[benchmark_input]
    stock_rets = returns.drop(columns=[benchmark_input], errors='ignore')

    # ==============================================================================
    # 3. QUANTITATIVE MODEL (CAPM REGRESSION)
    # ==============================================================================
    
    capm_data = []
    TRADING_DAYS = 252 
    rf_daily = rf_history_avg / TRADING_DAYS

    # Prepare Benchmark Excess Return (Keep NaNs for now)
    market_excess_series = bench_ret - rf_daily

    for ticker in stock_rets.columns:
        if ticker == benchmark_input: continue

        # Stock Excess Return
        stock_excess_series = stock_rets[ticker] - rf_daily

        # --- DATA ALIGNMENT & CLEANING (The Fix) ---
        # On combine le Benchmark et le Stock, puis on supprime les lignes 
        # où L'UN OU L'AUTRE est manquant.
        df_reg = pd.concat([market_excess_series, stock_excess_series], axis=1)
        df_reg.columns = ['Market', 'Stock']
        df_reg = df_reg.dropna() # ICI on nettoie proprement l'intersection

        # Security check: Need enough data points for regression
        if len(df_reg) < 30:
            # On passe silencieusement ou avec un warning léger
            continue

        x = df_reg['Market'].values
        y = df_reg['Stock'].values

        # Linear Regression
        try:
            beta, alpha_daily = np.polyfit(x, y, 1)
        except:
            continue
        
        # R-Squared
        r_squared = 0
        if len(x) > 2:
            correlation_matrix = np.corrcoef(x, y)
            correlation_xy = correlation_matrix[0,1]
            r_squared = correlation_xy**2

        # Annualize results
        alpha_annual = alpha_daily * TRADING_DAYS
        
        # Expected Return calculation
        # Note: We use the mean of the CLEANED benchmark data for consistency
        mkt_annual_ret = df_reg['Market'].mean() * TRADING_DAYS + rf_history_avg
        expected_return = rf_current + beta * (mkt_annual_ret - rf_current)
        
        actual_return = df_reg['Stock'].mean() * TRADING_DAYS + rf_history_avg

        capm_data.append({
            'Ticker': ticker,
            'Beta': beta,
            'Alpha (%)': alpha_annual,
            'Actual Return': actual_return,
            'Expected Return (CAPM)': expected_return,
            'R-Squared': r_squared,
            'Valuation': 'Undervalued' if alpha_annual > 0 else 'Overvalued'
        })

    # --- SAFETY CHECK: IF NO DATA ---
    if not capm_data:
        st.error("❌ No valid data found. This usually happens if the tickers are incorrect or if the analysis period is too short for the selected assets.")
        st.stop()

    df_capm = pd.DataFrame(capm_data).set_index('Ticker')

    # ==============================================================================
    # 4. VISUALIZATION DASHBOARD
    # ==============================================================================

    # --- KPI METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    # Recalculate global market metrics for display (using available data)
    mkt_return_ann = bench_ret.mean() * 252
    
    col1.metric("Benchmark", benchmark_input)
    col2.metric("Market Return (Ann.)", f"{mkt_return_ann:.1%}")
    col3.metric("Risk-Free Rate", f"{rf_current:.2%}")
    col4.metric("Market Risk Prem.", f"{(mkt_return_ann - rf_current):.1%}")

    st.markdown("---")

    col_chart, col_table = st.columns([2, 1])

    # --- CHART: SECURITY MARKET LINE (SML) ---
    with col_chart:
        st.subheader("Security Market Line (SML)")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # X-axis range
        max_beta = df_capm['Beta'].max()
        # Fallback if max_beta is weird
        if pd.isna(max_beta) or max_beta < 0.5: max_beta = 1.5
            
        x_range = np.linspace(0, max_beta * 1.2, 100)
        
        # SML Line
        y_sml = rf_current + x_range * (mkt_return_ann - rf_current)
        ax.plot(x_range, y_sml, color='#555555', linestyle='--', linewidth=2, label='SML (Fair Value)', alpha=0.8)

        # Plot Tickers
        colors = ['#228B22' if val == 'Undervalued' else '#D90429' for val in df_capm['Valuation']]
        
        ax.scatter(df_capm['Beta'], df_capm['Actual Return'], c=colors, s=100, zorder=5, edgecolors='white', linewidth=1)
        
        for ticker, row in df_capm.iterrows():
            ax.text(row['Beta'], row['Actual Return'] + 0.005, f"  {ticker}", fontsize=9, fontweight='bold')

        ax.set_xlabel('Beta (Systematic Risk)')
        ax.set_ylabel('Annualized Return')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper left')

        st.pyplot(fig, use_container_width=True)

    # --- TABLE: QUANTITATIVE DETAILS ---
    with col_table:
        st.subheader("Alpha Generation")
        
        df_display = df_capm[['Beta', 'Alpha (%)', 'Valuation']].copy()
        df_display = df_display.sort_values(by='Alpha (%)', ascending=False)
        
        df_display['Alpha (%)'] = df_display['Alpha (%)'].apply(lambda x: f"{x:.2%}")
        df_display['Beta'] = df_display['Beta'].apply(lambda x: f"{x:.2f}")

        st.dataframe(
            df_display,
            use_container_width=True,
            column_config={
                "Valuation": st.column_config.TextColumn(
                    "Signal",
                    help="Undervalued = Above SML",
                    width="medium"
                )
            }
        )

    st.markdown("---")

    with st.expander("📝 Methodological Note"):
        st.write("""
        **CAPM Implementation:**
        * **Data Alignment:** Returns are cleaned pairwise (Stock vs Benchmark) to maximize data availability.
        * **Risk-Free Rate:** Uses ^TNX (10Y Treasury).
        * **Alpha:** Jensen's Alpha (Annualized).
        """)