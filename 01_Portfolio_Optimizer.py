import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Project Portfolio - Julien Patron",
    layout="wide"
)
sns.set_theme(style="white", context="paper", font_scale=1.1)

# --- CSS POUR UN LOOK PRO ---
st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. BARRE LATÉRALE (INPUTS)
# ==============================================================================
st.sidebar.header("Configuration")

# A. Tickers
default_tickers = "LVMUY, TSM, JPM, PBR"
tickers_input = st.sidebar.text_input("Tickers (comma separated)", value=default_tickers)
tickers = [x.strip().upper() for x in tickers_input.split(',')]

# B. Target Return
target_input = st.sidebar.number_input("Target Annual Return (%)", min_value=0.0, max_value=100.0, value=12.0, step=0.5)
target_return = target_input / 100

# C. Time Horizon
years_back = st.sidebar.slider("Historical Data (Years)", min_value=1, max_value=10, value=5)

# D. Simulation Settings
n_simulations = 10000

# ==============================================================================
# 2. LOGIQUE PRINCIPALE
# ==============================================================================

st.title("Portfolio Optimizer")
st.markdown("Modern Portfolio Theory (Markowitz) & Capital Allocation Line.")
st.markdown("---")

# --- DATA PREP ---
with st.spinner('Processing market data...'):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=years_back*365)
    
    # Risk Free Rate (Dynamic)
    try:
        tnx = yf.Ticker("^TNX").history(period="5d")
        rf_rate = tnx['Close'].iloc[-1] / 100 if not tnx.empty else 0.04
    except:
        rf_rate = 0.04
        
    st.sidebar.markdown("---")
    st.sidebar.metric("Risk-Free Rate (10Y US Bond)", f"{rf_rate:.2%}")

    # Download Data
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna(axis=1, how='all')
        if data.shape[1] < 2:
            st.error("Error: Need at least 2 valid assets.")
            st.stop()
        
        tickers = data.columns.tolist()
        returns = data.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        num_assets = len(tickers)

    except Exception as e:
        st.error(f"Data Error: {e}")
        st.stop()

    # --- MONTE CARLO ENGINE ---
    sim_rets = np.zeros(n_simulations)
    sim_vols = np.zeros(n_simulations)
    sim_sharpes = np.zeros(n_simulations)
    sim_weights = np.zeros((n_simulations, num_assets))

    # Optimisation vectorisée
    weights = np.random.random((n_simulations, num_assets))
    weights /= np.sum(weights, axis=1)[:, np.newaxis]
    
    sim_rets = np.dot(weights, mean_returns.values)
    sim_vols = np.sqrt(np.diag(np.dot(weights, np.dot(cov_matrix.values, weights.T))))
    sim_sharpes = (sim_rets - rf_rate) / sim_vols

    # --- OPTIMIZATION ---
    max_idx = sim_sharpes.argmax()
    tan_ret = sim_rets[max_idx]
    tan_vol = sim_vols[max_idx]
    tan_sharpe = sim_sharpes[max_idx]
    tan_weights = weights[max_idx]

    # --- ALLOCATION ---
    if target_return < rf_rate:
        w_invest = 0
        client_vol = 0
    else:
        w_invest = (target_return - rf_rate) / (tan_ret - rf_rate)
        client_vol = w_invest * tan_vol
    w_cash = 1 - w_invest

    # ==============================================================================
    # 3. AFFICHAGE DES RÉSULTATS
    # ==============================================================================
    
    # --- SECTION 1: STRATEGIC REPORT (Moved to Top) ---
    st.subheader("1. Allocation Report")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Risk-Free Asset (10Y US Bond)", f"{w_cash:.1%}")
    col2.metric("Equity Allocation", f"{w_invest:.1%}")
    col3.metric("Expected Return (Target)", f"{target_return:.1%}")
    col4.metric("Est. Annual Volatility", f"{client_vol:.1%}")

    st.write("#### Equity Composition")
    df_final = pd.DataFrame({
        "Ticker": tickers,
        "Tangency Weight": tan_weights,
        "Final Portfolio Weight": tan_weights * w_invest
    }).sort_values(by="Final Portfolio Weight", ascending=False)
    
    df_display = df_final.copy()
    df_display["Tangency Weight"] = df_display["Tangency Weight"].apply(lambda x: f"{x:.1%}")
    df_display["Final Portfolio Weight"] = df_display["Final Portfolio Weight"].apply(lambda x: f"{x:.1%}")
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- SECTION 2: EFFICIENT FRONTIER (Moved below) ---
    st.subheader("2. Efficient Frontier & Capital Allocation")
    
    # On crée un figure plus large (width=12) pour qu'elle prenne toute la place
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Cloud
    sc = ax.scatter(sim_vols, sim_rets, c=sim_sharpes, cmap='RdYlGn', s=3, alpha=0.5, rasterized=True)
    
    # Scale fix
    stock_vols = np.sqrt(np.diag(cov_matrix))
    stock_rets = mean_returns.values
    max_stock_vol = np.max(stock_vols)
    max_x = max(tan_vol, client_vol, max_stock_vol) * 1.15
    
    # CAL
    ax.plot([0, max_x], [rf_rate, rf_rate + tan_sharpe * max_x], color='#555555', linestyle='--', linewidth=1, alpha=0.8, label='Capital Allocation Line')
    
    # Points
    ax.scatter(0, rf_rate, c='black', marker='_', s=200, linewidth=2)
    ax.text(0.002, rf_rate, f'Risk-Free ({rf_rate:.1%})', fontsize=9, fontweight='bold', va='bottom')

    ax.scatter(tan_vol, tan_ret, c='#D90429', marker='+', s=150, linewidth=1.5, zorder=10)
    ax.text(tan_vol, tan_ret+0.005, 'Tangency Portfolio', fontsize=9, fontweight='bold', color='#D90429', ha='right', va='bottom')

    ax.scatter(client_vol, target_return, c='#0077B6', marker='x', s=100, linewidth=1.5, zorder=10)
    ax.text(client_vol, target_return+0.005, f'Target ({target_return:.1%})', fontsize=9, fontweight='bold', color='#0077B6', ha='right', va='bottom')

    # Stocks
    ax.scatter(stock_vols, stock_rets, c='black', marker='x', s=50, zorder=15)
    for i, txt in enumerate(tickers):
        ax.text(stock_vols[i], stock_rets[i]+0.005, f' {txt}', fontsize=9, fontweight='bold', va='bottom')

    # Format
    ax.set_xlabel('Annualized Volatility (Risk)')
    ax.set_ylabel('Annualized Return')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, max_x)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)
    
    # Legend & Colorbar
    ax.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.95, edgecolor='#E0E0E0')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Sharpe Ratio')
    
    # use_container_width=True force le graphique à prendre toute la largeur
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    # --- SECTION 3: METRICS & CORRELATION ---
    col_metrics, col_corr = st.columns([1, 1])

    with col_metrics:
        st.subheader("3. Individual Asset Metrics")
        stock_vols_series = returns.std() * np.sqrt(252)
        df_metrics = pd.DataFrame({
            'Return': mean_returns,
            'Volatility': stock_vols_series
        })
        # Format
        df_metrics_disp = df_metrics.copy()
        df_metrics_disp['Return'] = df_metrics_disp['Return'].apply(lambda x: f"{x:.1%}")
        df_metrics_disp['Volatility'] = df_metrics_disp['Volatility'].apply(lambda x: f"{x:.1%}")
        st.dataframe(df_metrics_disp, use_container_width=True)

    with col_corr:
        st.subheader("4. Correlation Matrix")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1, ax=ax_corr)
        st.pyplot(fig_corr)