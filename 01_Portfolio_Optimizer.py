import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import datetime

# --- AESTHETIC SETUP ---
sns.set_theme(style="white", context="paper", font_scale=1.1)

# ==============================================================================
# 1. USER INTERFACE & CONFIGURATION
# ==============================================================================

# Configuration de la page en mode "Wide" (Large) par défaut
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

print("--- PORTFOLIO OPTIMIZER ---")

# A. Tickers Input
default_tickers_str = "TM, NVS, TSM, BP"
tickers_input = st.sidebar.text_input("Tickers (comma separated)", value=default_tickers_str)

if tickers_input.strip() == "":
    tickers = [x.strip() for x in default_tickers_str.split(',')]
else:
    tickers = [x.strip().upper() for x in tickers_input.split(',')]

# B. Target Return Input
default_return = 12.0
target_input = st.sidebar.number_input("Target Annual Return (%)", min_value=0.0, max_value=100.0, value=default_return, step=0.5)
target_return = target_input / 100

# C. Time Horizon Input
default_years = 5
years_back = st.sidebar.slider("Historical Data (Years)", min_value=1, max_value=10, value=default_years)

# Dates
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=years_back*365)
start_str = start_date.strftime("%Y-%m-%d")
end_str = end_date.strftime("%Y-%m-%d")

# D. Risk Free Rate
try:
    tnx = yf.Ticker("^TNX")
    tnx_hist = tnx.history(period="5d")
    if not tnx_hist.empty:
        rf_rate = tnx_hist['Close'].iloc[-1] / 100
    else:
        rf_rate = 0.04
except:
    rf_rate = 0.04

st.sidebar.caption(f"Risk-Free Rate (10Y US): {rf_rate:.2%}")

n_simulations = 10000

# ==============================================================================
# 2. DATA PROCESSING & ENGINE
# ==============================================================================

st.title("Strategic Portfolio Optimizer")
st.markdown("---")

# Utilisation d'un spinner pour le chargement
with st.spinner('Downloading market data and running simulation...'):
    try:
        data_raw = yf.download(tickers, start=start_str, end=end_str)['Close']
        data = data_raw.dropna(axis=1, how='all')
        
        if data.shape[1] < 2:
            st.error("Error: Need at least 2 valid assets.")
            st.stop()
            
        tickers = data.columns.tolist()
        num_assets = len(tickers)

        # Financial Math
        returns = data.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # Monte Carlo
        sim_rets = np.zeros(n_simulations)
        sim_vols = np.zeros(n_simulations)
        sim_sharpes = np.zeros(n_simulations)
        
        # Vectorized Optimization (Ultra Fast)
        weights = np.random.random((n_simulations, num_assets))
        weights /= np.sum(weights, axis=1)[:, np.newaxis]
        
        sim_rets = np.dot(weights, mean_returns.values)
        sim_vols = np.sqrt(np.diag(np.dot(weights, np.dot(cov_matrix.values, weights.T))))
        sim_sharpes = (sim_rets - rf_rate) / sim_vols

        # Tangency
        max_idx = sim_sharpes.argmax()
        tan_ret = sim_rets[max_idx]
        tan_vol = sim_vols[max_idx]
        tan_sharpe = sim_sharpes[max_idx]
        tan_weights = weights[max_idx]

        # Capital Allocation
        if target_return < rf_rate:
            w_invest = 0
            client_vol = 0
        else:
            w_invest = (target_return - rf_rate) / (tan_ret - rf_rate)
            client_vol = w_invest * tan_vol
        w_cash = 1 - w_invest

    except Exception as e:
        st.error(f"Critical Error: {e}")
        st.stop()

# ==============================================================================
# 3. DISPLAY RESULTS
# ==============================================================================

# --- PART 1: STRATEGIC REPORT ---
st.subheader(f"1. Allocation Report (Target: {target_return:.1%})")

c1, c2, c3 = st.columns(3)
c1.metric("Risk-Free Asset (Cash)", f"{w_cash:.1%}")
c2.metric("Equity Portfolio", f"{w_invest:.1%}")
c3.metric("Est. Annual Volatility", f"{client_vol:.1%}")

st.write("**Equity Composition:**")
df_alloc = pd.DataFrame({
    'Asset': tickers,
    'Tangency Weight': tan_weights,
    'Final Weight': tan_weights * w_invest
}).sort_values(by='Final Weight', ascending=False)

df_disp = df_alloc.copy()
df_disp['Tangency Weight'] = df_disp['Tangency Weight'].apply(lambda x: f"{x:.1%}")
df_disp['Final Weight'] = df_disp['Final Weight'].apply(lambda x: f"{x:.1%}")
st.dataframe(df_disp, use_container_width=True, hide_index=True)

st.markdown("---")

# --- PART 2: VISUALIZATION (FULL WIDTH) ---
st.subheader("2. Efficient Frontier & Capital Allocation")

# Create figure with wider ratio (14:6)
fig, ax = plt.subplots(figsize=(14, 6))

# A. The Cloud
sc = ax.scatter(sim_vols, sim_rets, c=sim_sharpes, cmap='RdYlGn', s=3, alpha=0.5, rasterized=True)

# B. The CAL
stock_vols = np.sqrt(np.diag(cov_matrix))
stock_rets = mean_returns.values
max_stock_vol = np.max(stock_vols)
max_plot_x = max(tan_vol, client_vol, max_stock_vol) * 1.15

cal_x = [0, max_plot_x]
cal_y = [rf_rate, rf_rate + tan_sharpe * max_plot_x]
ax.plot(cal_x, cal_y, color='#555555', linestyle='--', linewidth=1, alpha=0.8, label='Capital Allocation Line')

# C. Key Points
ax.scatter(0, rf_rate, c='black', marker='_', s=200, linewidth=2)
ax.text(0.002, rf_rate, f'Risk-Free ({rf_rate:.1%})', fontsize=9, fontweight='bold', va='bottom')

ax.scatter(tan_vol, tan_ret, c='#D90429', marker='+', s=150, linewidth=1.5, zorder=10)
ax.text(tan_vol, tan_ret+0.005, 'Tangency', fontsize=9, fontweight='bold', color='#D90429', ha='right', va='bottom')

ax.scatter(client_vol, target_return, c='#0077B6', marker='x', s=100, linewidth=1.5, zorder=10)
ax.text(client_vol, target_return+0.005, 'Target', fontsize=9, fontweight='bold', color='#0077B6', ha='right', va='bottom')

# Stocks
ax.scatter(stock_vols, stock_rets, c='black', marker='x', s=50, zorder=15)
for i, txt in enumerate(tickers):
    ax.text(stock_vols[i], stock_rets[i]+0.005, f' {txt}', fontsize=9, fontweight='bold', va='bottom')

# D. Formatting
ax.set_xlabel('Annualized Volatility (Risk)')
ax.set_ylabel('Annualized Return')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.set_xlim(0, max_plot_x)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.2)

# Legend & Colorbar
ax.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.95, edgecolor='#E0E0E0')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Sharpe Ratio')

# DISPLAY FULL WIDTH
st.pyplot(fig, use_container_width=True)

st.markdown("---")

# --- PART 3: ANALYSIS (Two Columns) ---
col_metrics, col_corr = st.columns([1, 1])

with col_metrics:
    st.subheader("3. Asset Metrics")
    df_metrics = pd.DataFrame({
        'Return': mean_returns,
        'Volatility': stock_vols
    })
    df_metrics_disp = df_metrics.copy()
    df_metrics_disp['Return'] = df_metrics_disp['Return'].apply(lambda x: f"{x:.1%}")
    df_metrics_disp['Volatility'] = df_metrics_disp['Volatility'].apply(lambda x: f"{x:.1%}")
    st.dataframe(df_metrics_disp, use_container_width=True)

with col_corr:
    st.subheader("4. Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1, ax=ax_corr)
    st.pyplot(fig_corr, use_container_width=True)
