import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Project Portfolio - Julien Patron",
    layout="wide"
)

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
n_simulations = 5000 # Réduit légèrement pour la vitesse, suffisant pour la démo

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
    
    # Risk Free Rate (Optimisé: valeur par défaut rapide si échec)
    rf_rate = 0.04
    try:
        tnx = yf.Ticker("^TNX").history(period="5d")
        if not tnx.empty:
            rf_rate = tnx['Close'].iloc[-1] / 100
    except:
        pass
        
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

    # --- MONTE CARLO ENGINE (Optimisé numpy) ---
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
    
    # --- SECTION 1: STRATEGIC REPORT ---
    st.subheader("1. Allocation Report")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk-Free Asset", f"{w_cash:.1%}")
    col2.metric("Equity Allocation", f"{w_invest:.1%}")
    col3.metric("Expected Return", f"{target_return:.1%}")
    col4.metric("Est. Volatility", f"{client_vol:.1%}")

    st.write("#### Equity Composition")
    df_final = pd.DataFrame({
        "Ticker": tickers,
        "Tangency Weight": tan_weights,
        "Final Portfolio Weight": tan_weights * w_invest
    }).sort_values(by="Final Portfolio Weight", ascending=False)
    
    st.dataframe(
        df_final.style.format("{:.1%}"), 
        use_container_width=True, 
        hide_index=True
    )

    st.markdown("---")

    # --- SECTION 2: EFFICIENT FRONTIER (PLOTLY VERSION) ---
    st.subheader("2. Efficient Frontier & Capital Allocation")
    
    # Préparation des données pour Plotly
    # On limite le nombre de points affichés pour la perf si nécessaire, mais 5000 ça passe large
    
    fig = go.Figure()

    # 1. Nuage de points (Monte Carlo)
    fig.add_trace(go.Scatter(
        x=sim_vols, y=sim_rets,
        mode='markers',
        marker=dict(
            size=5,
            color=sim_sharpes,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        name='Simulations',
        text=[f"Vol: {v:.1%}<br>Ret: {r:.1%}" for v, r in zip(sim_vols, sim_rets)],
        hoverinfo='text'
    ))

    # 2. Capital Allocation Line (CAL)
    # Calcul de la ligne max
    max_vol_display = max(sim_vols.max(), client_vol) * 1.2
    cal_x = [0, max_vol_display]
    cal_y = [rf_rate, rf_rate + tan_sharpe * max_vol_display]
    
    fig.add_trace(go.Scatter(
        x=cal_x, y=cal_y,
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        name='Capital Allocation Line'
    ))

    # 3. Points clés (Risk Free, Tangency, Target)
    fig.add_trace(go.Scatter(
        x=[0], y=[rf_rate],
        mode='markers+text',
        marker=dict(color='black', size=12, symbol='diamond'),
        text=[f"Risk-Free<br>{rf_rate:.1%}"],
        textposition="bottom right",
        name='Risk-Free'
    ))

    fig.add_trace(go.Scatter(
        x=[tan_vol], y=[tan_ret],
        mode='markers+text',
        marker=dict(color='red', size=15, symbol='star'),
        text=["Tangency Portfolio"],
        textposition="top left",
        name='Tangency'
    ))

    fig.add_trace(go.Scatter(
        x=[client_vol], y=[target_return],
        mode='markers+text',
        marker=dict(color='blue', size=12, symbol='x'),
        text=[f"Target<br>{target_return:.1%}"],
        textposition="bottom right",
        name='Target'
    ))

    # 4. Actifs individuels
    stock_vols = np.sqrt(np.diag(cov_matrix))
    stock_rets = mean_returns.values
    fig.add_trace(go.Scatter(
        x=stock_vols, y=stock_rets,
        mode='markers+text',
        marker=dict(color='black', size=8),
        text=tickers,
        textposition="top center",
        name='Assets'
    ))

    fig.update_layout(
        xaxis_title="Annualized Volatility (Risk)",
        yaxis_title="Annualized Return",
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickformat=".0%"),
        height=600,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.5)')
    )

    st.plotly_chart(fig, use_container_width=True)

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
        st.dataframe(df_metrics.style.format("{:.1%}"), use_container_width=True)

    with col_corr:
        st.subheader("4. Correlation Matrix")
        # Remplacement de Seaborn par Plotly Heatmap (plus léger)
        corr = returns.corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            showscale=False
        ))
        fig_corr.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)