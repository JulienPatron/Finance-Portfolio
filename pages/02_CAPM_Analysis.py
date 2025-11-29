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
tickers = [x.strip().upper() for x in tickers_input.split(',')]

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

    # 1. Get Risk Free Rate Data (History for regression, Spot for SML)
    try:
        tnx = yf.download("^TNX", start=start_date, end=end_date, progress=False)['Close']
        rf_series = tnx / 100  # Convert to decimal
        rf_current = rf_series.iloc[-1]
        rf_history_avg = rf_series.mean()
    except:
        # Fallback if Yahoo Finance fails on TNX
        rf_current = 0.04
        rf_history_avg = 0.04
        st.warning("Could not fetch ^TNX data. Using static 4.0% Risk-Free Rate.")

    # 2. Download Stock & Benchmark Data
    all_symbols = tickers + [benchmark_input]
    try:
        data = yf.download(all_symbols, start=start_date, end=end_date, progress=False)['Close']
        
        # Data Cleaning
        data = data.dropna(axis=1, how='all').dropna()
        
        if benchmark_input not in data.columns:
            st.error(f"Benchmark '{benchmark_input}' data not found. Please check the symbol.")
            st.stop()
            
    except Exception as e:
        st.error(f"Data Download Error: {e}")
        st.stop()

    # 3. Calculate Returns (Arithmetic)
    returns = data.pct_change().dropna()
    
    # Separate Benchmark and Stocks
    bench_ret = returns[benchmark_input]
    stock_rets = returns.drop(columns=[benchmark_input])

    # ==============================================================================
    # 3. QUANTITATIVE MODEL (CAPM REGRESSION)
    # ==============================================================================
    
    capm_data = []

    # Annualize factor for Alpha/Return
    TRADING_DAYS = 252 

    # Prepare Benchmark Excess Return for Regression (X axis)
    # We use the historical average Rf for the regression period stability
    rf_daily = rf_history_avg / TRADING_DAYS
    market_excess = bench_ret - rf_daily

    for ticker in stock_rets.columns:
        # 1. Prepare inputs (Excess Returns)
        y_raw = stock_rets[ticker] - rf_daily
        x_raw = market_excess

        # 2. DATA CLEANING & ALIGNMENT (CRITICAL STEP)
        # On combine les deux séries dans un DataFrame temporaire pour supprimer 
        # les lignes où l'une des deux valeurs est NaN.
        df_reg = pd.concat([x_raw, y_raw], axis=1).dropna()
        df_reg.columns = ['Market', 'Stock']
        
        # Security check: Need enough data points for regression
        if len(df_reg) < 30:
            st.warning(f"Skipping {ticker}: Not enough data points aligned with Benchmark.")
            continue

        x = df_reg['Market'].values
        y = df_reg['Stock'].values

        # 3. Linear Regression (Polyfit degree 1)
        # Slope = Beta, Intercept = Daily Alpha
        try:
            beta, alpha_daily = np.polyfit(x, y, 1)
        except Exception as e:
            st.warning(f"Could not calculate Beta for {ticker}: {e}")
            continue
        
        # Calculate R-Squared
        correlation_matrix = np.corrcoef(x, y)
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2

        # Annualize Alpha
        alpha_annual = alpha_daily * TRADING_DAYS

        # Expected Return (CAPM Theory)
        mkt_annual_ret = bench_ret.mean() * TRADING_DAYS
        expected_return = rf_current + beta * (mkt_annual_ret - rf_current)
        
        # Actual Annualized Return
        actual_return = stock_rets[ticker].mean() * TRADING_DAYS

        capm_data.append({
            'Ticker': ticker,
            'Beta': beta,
            'Alpha (%)': alpha_annual,
            'Actual Return': actual_return,
            'Expected Return (CAPM)': expected_return,
            'R-Squared': r_squared,
            'Valuation': 'Undervalued' if alpha_annual > 0 else 'Overvalued'
        })

    df_capm = pd.DataFrame(capm_data).set_index('Ticker')

    # ==============================================================================
    # 4. VISUALIZATION DASHBOARD
    # ==============================================================================

    # --- KPI METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    mkt_return_ann = bench_ret.mean() * 252
    
    col1.metric("Benchmark", benchmark_input)
    col2.metric("Market Return (Ann.)", f"{mkt_return_ann:.1%}")
    col3.metric("Risk-Free Rate (Current)", f"{rf_current:.2%}")
    col4.metric("Market Risk Prem. (MRP)", f"{(mkt_return_ann - rf_current):.1%}")

    st.markdown("---")

    col_chart, col_table = st.columns([2, 1])

    # --- CHART: SECURITY MARKET LINE (SML) ---
    with col_chart:
        st.subheader("Security Market Line (SML)")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 1. Plot SML Line (Theoretical)
        # X-axis range: from 0 to max beta + margin
        max_beta = df_capm['Beta'].max()
        x_range = np.linspace(0, max_beta * 1.2, 100)
        
        # SML Equation: y = Rf + x * (Rm - Rf)
        # Using CURRENT Rf and Historical MRP for the "Forward looking" line
        y_sml = rf_current + x_range * (mkt_return_ann - rf_current)
        
        ax.plot(x_range, y_sml, color='#555555', linestyle='--', linewidth=2, label='SML (Fair Value)', alpha=0.8)

        # 2. Plot Tickers
        # Color coding based on Alpha
        colors = ['#228B22' if val == 'Undervalued' else '#D90429' for val in df_capm['Valuation']]
        
        ax.scatter(df_capm['Beta'], df_capm['Actual Return'], c=colors, s=100, zorder=5, edgecolors='white', linewidth=1)
        
        # Labels
        for ticker, row in df_capm.iterrows():
            ax.text(row['Beta'], row['Actual Return'] + 0.005, f"  {ticker}", fontsize=9, fontweight='bold')

        # Formatting
        ax.set_xlabel('Beta (Systematic Risk)')
        ax.set_ylabel('Annualized Actual Return')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper left')

        # Annotation Helper
        ax.text(0.1, y_sml[-1], "Undervalued (Buy)\n(Alpha > 0)", color='#228B22', fontsize=8, ha='left')
        ax.text(max_beta, rf_current, "Overvalued (Sell)\n(Alpha < 0)", color='#D90429', fontsize=8, ha='right')

        st.pyplot(fig, use_container_width=True)

    # --- TABLE: QUANTITATIVE DETAILS ---
    with col_table:
        st.subheader("Alpha Generation")
        
        # Formatting for display
        df_display = df_capm[['Beta', 'Alpha (%)', 'Valuation']].copy()
        df_display = df_display.sort_values(by='Alpha (%)', ascending=False)
        
        # Format percentages
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

    # --- EXPLANATION SECTION ---
    with st.expander("📝 Methodological Note (For Recruiters)"):
        st.write("""
        **Capital Asset Pricing Model (CAPM) Implementation:**
        
        1.  **Data Source:** Yahoo Finance (Adjusted Close).
        2.  **Risk-Free Rate ($R_f$):** Uses the 10-Year US Treasury Yield (`^TNX`). 
            * *Calculation:* Historical average used for regression intercepts; Current spot rate used for SML construction.
        3.  **Beta ($\beta$):** Calculated via Linear Regression of Stock Excess Returns vs. Benchmark Excess Returns.
        4.  **Alpha ($\alpha$):** Jensen's Alpha (Intercept). Represents the annualized return in excess of the theoretical CAPM return.
            * $$ \alpha = R_p - [R_f + \beta_p (R_m - R_f)] $$
        """)