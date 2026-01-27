import streamlit as st
import pandas as pd
import numpy as np
import datetime

# Note : Pas de set_page_config ici car c'est géré par main.py

# ==============================================================================
# 1. FONCTIONS DE CHARGEMENT & CALCUL (BACKEND)
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data(tickers, years_back):
    """
    Télécharge les données boursières et le taux sans risque.
    Mise en cache 1h pour éviter de spammer Yahoo Finance.
    Import lazy de yfinance ici pour ne pas ralentir le démarrage de l'app.
    """
    import yfinance as yf
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=years_back*365)

    # 1. Risk Free Rate (Taux sans risque 10 ans US)
    try:
        tnx = yf.Ticker("^TNX").history(period="5d")
        rf_rate = tnx['Close'].iloc[-1] / 100 if not tnx.empty else 0.04
    except:
        rf_rate = 0.04

    # 2. Données Actions
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        # Nettoyage des colonnes vides
        data = data.dropna(axis=1, how='all')
        return rf_rate, data
    except Exception as e:
        return rf_rate, None

# ==============================================================================
# 2. BARRE LATÉRALE (INPUTS)
# ==============================================================================
st.sidebar.header("Configuration")

# A. Tickers
default_tickers = "LVMUY, TSM, JPM, PBR"
tickers_input = st.sidebar.text_input("Tickers (séparés par virgule)", value=default_tickers)
tickers_list = [x.strip().upper() for x in tickers_input.split(',') if x.strip()]

# B. Target Return
target_input = st.sidebar.number_input("Target Annual Return (%)", min_value=0.0, max_value=100.0, value=12.0, step=0.5)
target_return = target_input / 100

# C. Time Horizon
years_back = st.sidebar.slider("Historical Data (Years)", min_value=1, max_value=10, value=5)

# D. Simulation Settings
n_simulations = 5000 # Réduit légèrement pour la réactivité du Free Tier, suffisant pour converger

# ==============================================================================
# 3. LOGIQUE PRINCIPALE
# ==============================================================================

st.title("Portfolio Optimizer")
st.markdown("Modern Portfolio Theory (Markowitz) & Capital Allocation Line.")
st.markdown("---")

# --- DATA PREP ---
if not tickers_list:
    st.warning("Veuillez entrer des tickers valides.")
    st.stop()

with st.spinner('Processing market data...'):
    rf_rate, data = get_market_data(tickers_list, years_back)

    if data is None or data.shape[1] < 2:
        st.error("Erreur : Il faut au moins 2 actifs valides trouvés sur Yahoo Finance.")
        st.stop()

    # Calculs Financiers
    tickers = data.columns.tolist()
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = len(tickers)

    # Affichage RF Rate dans la sidebar
    st.sidebar.markdown("---")
    st.sidebar.metric("Risk-Free Rate (10Y US Bond)", f"{rf_rate:.2%}")

    # --- MONTE CARLO ENGINE (Vectorisé) ---
    # Génération aléatoire des poids
    weights = np.random.random((n_simulations, num_assets))
    weights /= np.sum(weights, axis=1)[:, np.newaxis]

    # Calcul des métriques de portefeuille (Matriciel = Rapide)
    sim_rets = np.dot(weights, mean_returns.values)
    sim_vols = np.sqrt(np.diag(np.dot(weights, np.dot(cov_matrix.values, weights.T))))
    sim_sharpes = (sim_rets - rf_rate) / sim_vols

    # --- OPTIMIZATION ---
    max_idx = sim_sharpes.argmax()
    tan_ret = sim_rets[max_idx]
    tan_vol = sim_vols[max_idx]
    tan_sharpe = sim_sharpes[max_idx]
    tan_weights = weights[max_idx]

    # --- ALLOCATION (CAL) ---
    if target_return < rf_rate:
        w_invest = 0.0
        client_vol = 0.0
    else:
        # Ratio de levier ou d'exposition nécessaire
        w_invest = (target_return - rf_rate) / (tan_ret - rf_rate) if (tan_ret - rf_rate) != 0 else 0
        client_vol = w_invest * tan_vol
    
    w_cash = 1.0 - w_invest

    # ==============================================================================
    # 4. AFFICHAGE DES RÉSULTATS
    # ==============================================================================

    # --- SECTION 1: STRATEGIC REPORT ---
    st.subheader("1. Allocation Report")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk-Free Asset", f"{w_cash:.1%}")
    col2.metric("Equity Allocation", f"{w_invest:.1%}")
    col3.metric("Target Return", f"{target_return:.1%}")
    col4.metric("Est. Volatility", f"{client_vol:.1%}")

    st.write("#### Equity Composition")
    
    # Création du DataFrame de poids
    df_final = pd.DataFrame({
        "Ticker": tickers,
        "Tangency Weight": tan_weights,
        "Final Portfolio Weight": tan_weights * w_invest
    }).sort_values(by="Final Portfolio Weight", ascending=False)
    
    # Formatage pour l'affichage (String %)
    df_display = df_final.copy()
    df_display["Tangency Weight"] = df_display["Tangency Weight"].apply(lambda x: f"{x:.1%}")
    df_display["Final Portfolio Weight"] = df_display["Final Portfolio Weight"].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- SECTION 2: EFFICIENT FRONTIER (PLOTLY) ---
    # Import Lazy de Plotly uniquement maintenant
    import plotly.graph_objects as go
    import plotly.express as px

    st.subheader("2. Efficient Frontier & Capital Allocation")

    # Création de la figure
    fig = go.Figure()

    # 1. Le Nuage de points (Monte Carlo) - Scattergl pour la perf
    fig.add_trace(go.Scattergl(
        x=sim_vols, 
        y=sim_rets,
        mode='markers',
        marker=dict(
            color=sim_sharpes, 
            colorscale='RdYlGn', 
            size=4, 
            opacity=0.5,
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        name='Simulations'
    ))

    # 2. Capital Allocation Line (CAL)
    max_stock_vol = np.max(np.sqrt(np.diag(cov_matrix)))
    max_x = max(tan_vol, client_vol, max_stock_vol) * 1.2
    
    fig.add_trace(go.Scatter(
        x=[0, max_x],
        y=[rf_rate, rf_rate + tan_sharpe * max_x],
        mode='lines',
        line=dict(color='gray', dash='dash', width=1),
        name='Capital Allocation Line'
    ))

    # 3. Points Spécifiques (Risk Free, Tangency, Target)
    # Risk Free
    fig.add_trace(go.Scatter(
        x=[0], y=[rf_rate],
        mode='markers+text',
        marker=dict(color='black', size=10, symbol='circle-open'),
        text=["Risk Free"], textposition="top right",
        name='Risk Free'
    ))

    # Tangency Portfolio
    fig.add_trace(go.Scatter(
        x=[tan_vol], y=[tan_ret],
        mode='markers+text',
        marker=dict(color='#D90429', size=12, symbol='cross'),
        text=["Tangency"], textposition="bottom right",
        name='Tangency Portfolio'
    ))

    # Target Portfolio
    fig.add_trace(go.Scatter(
        x=[client_vol], y=[target_return],
        mode='markers+text',
        marker=dict(color='#0077B6', size=12, symbol='x'),
        text=["Target"], textposition="top left",
        name='Target Portfolio'
    ))

    # 4. Actions individuelles
    stock_vols_series = returns.std() * np.sqrt(252)
    fig.add_trace(go.Scatter(
        x=stock_vols_series,
        y=mean_returns,
        mode='markers+text',
        marker=dict(color='black', size=8, symbol='diamond'),
        text=tickers, textposition="top center",
        name='Assets'
    ))

    # Mise en forme du Layout
    fig.update_layout(
        xaxis_title="Annualized Volatility (Risk)",
        yaxis_title="Annualized Return",
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickformat=".0%"),
        height=600,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center")
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- SECTION 3: METRICS & CORRELATION ---
    col_metrics, col_corr = st.columns([1, 1])

    with col_metrics:
        st.subheader("3. Asset Metrics")
        df_metrics = pd.DataFrame({
            'Return': mean_returns,
            'Volatility': stock_vols_series
        })
        # Formatage
        df_metrics_disp = df_metrics.copy()
        df_metrics_disp['Return'] = df_metrics_disp['Return'].apply(lambda x: f"{x:.1%}")
        df_metrics_disp['Volatility'] = df_metrics_disp['Volatility'].apply(lambda x: f"{x:.1%}")
        st.dataframe(df_metrics_disp, use_container_width=True)

    with col_corr:
        st.subheader("4. Correlation Matrix")
        # Heatmap interactive avec Plotly Express
        fig_corr = px.imshow(
            returns.corr(),
            text_auto=".2f",
            color_continuous_scale='RdBu_r', # Red Blue reverse (Rouge = corr negative, Bleu = positive)
            zmin=-1, zmax=1,
            aspect="auto"
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)