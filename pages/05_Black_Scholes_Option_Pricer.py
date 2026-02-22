import streamlit as st
import numpy as np
import scipy.stats as si
import plotly.graph_objects as go
import yfinance as yf

# --- CONFIGURATION DE LA PAGE ---
# st.set_page_config(page_title="Black-Scholes Pricer", layout="wide") # Si tu l'utilises seul. Sinon, géré par ton main.py

# ==============================================================================
# 1. MOTEUR MATHÉMATIQUE (BLACK-SCHOLES)
# ==============================================================================

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Calcule le prix d'une option européenne avec le modèle de Black-Scholes."""
    # Gestion du cas T=0 (Échéance)
    if T <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    
    # Calcul de d1 et d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)

# ==============================================================================
# 2. RÉCUPÉRATION DES DONNÉES (API YAHOO FINANCE)
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(ticker_symbol):
    """Récupère le prix en direct, la vol historique et le taux sans risque."""
    try:
        # 1. Sous-jacent
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        if hist.empty:
            return None
        current_price = float(hist['Close'].iloc[-1])
        
        # 2. Volatilité Historique (écart-type des rendements log quotidiens * sqrt(252))
        hist['Log_Ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
        volatility = float(hist['Log_Ret'].std() * np.sqrt(252))
        
        # 3. Taux sans risque (Bons du Trésor US à 10 ans)
        tnx = yf.Ticker("^TNX")
        tnx_hist = tnx.history(period="5d")
        if not tnx_hist.empty:
            rf_rate = float(tnx_hist['Close'].iloc[-1]) / 100.0
        else:
            rf_rate = 0.04 # Fallback à 4%
            
        return {"price": current_price, "vol": volatility, "rf_rate": rf_rate}
    except Exception as e:
        return None

# ==============================================================================
# 3. GESTION DES VARIABLES D'ÉTAT (SESSION STATE)
# ==============================================================================

# Initialisation des valeurs par défaut si elles n'existent pas
if 'S' not in st.session_state: st.session_state['S'] = 100.0
if 'K' not in st.session_state: st.session_state['K'] = 100.0
if 'vol' not in st.session_state: st.session_state['vol'] = 0.20
if 'rf' not in st.session_state: st.session_state['rf'] = 0.04

# ==============================================================================
# 4. INTERFACE UTILISATEUR (SIDEBAR & INPUTS)
# ==============================================================================

st.title("📈 Black-Scholes Option Pricing & Analytics")

with st.sidebar:
    st.header("1. Market Data (API)")
    ticker = st.text_input("Ticker Yahoo Finance (ex: AAPL, TSLA)", "AAPL")
    
    if st.button("Fetch Live Data", type="primary"):
        with st.spinner("Fetching data..."):
            data = fetch_market_data(ticker)
            if data:
                st.session_state['S'] = data['price']
                st.session_state['K'] = round(data['price'], 2) # Strike par défaut = prix actuel
                st.session_state['vol'] = data['vol']
                st.session_state['rf'] = data['rf_rate']
                st.success("Data updated!")
            else:
                st.error("Ticker not found.")
                
    st.divider()
    
    st.header("2. Manual Overrides")
    st.caption("Ajustez ces valeurs pour simuler vos propres scénarios.")
    
    # Inputs manuels (liés au Session State pour se mettre à jour avec l'API)
    S = st.number_input("Stock Price ($)", min_value=1.0, value=st.session_state['S'], step=1.0)
    K = st.number_input("Strike Price ($)", min_value=1.0, value=st.session_state['K'], step=1.0)
    
    # Sliders
    days_to_maturity = st.slider("Days to Maturity", min_value=1, max_value=365, value=30)
    T = days_to_maturity / 365.0 # Le modèle de B&S utilise le temps en années
    
    sigma = st.slider("Volatility (σ)", min_value=0.01, max_value=1.50, value=float(st.session_state['vol']), step=0.01)
    r = st.slider("Risk-Free Rate (r)", min_value=0.00, max_value=0.15, value=float(st.session_state['rf']), step=0.01)

# ==============================================================================
# 5. CALCULS EN DIRECT & KPI DASHBOARD
# ==============================================================================

# Calcul des prix des options
call_price = black_scholes(S, K, T, r, sigma, "call")
put_price = black_scholes(S, K, T, r, sigma, "put")

st.subheader("Pricing Dashboard")

# Affichage des KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Underlying Price", f"${S:,.2f}")
col2.metric("Call Premium", f"${call_price:,.2f}")
col3.metric("Put Premium", f"${put_price:,.2f}")
col4.metric("Time to Expiry", f"{days_to_maturity} Days")

st.divider()

# ==============================================================================
# 6. GRAPHIQUES (PAYOFF & HEATMAPS)
# ==============================================================================

tab_payoff, tab_heatmap = st.tabs(["PnL & Payoff at Expiration", "Pricing Heatmaps (Live)"])

with tab_payoff:
    st.subheader("Profit & Loss at Expiration")
    
    # Création d'un tableau de prix possibles à l'échéance (de -30% à +30% du prix actuel)
    S_range_1d = np.linspace(S * 0.7, S * 1.3, 100)
    
    # Calcul du Payoff et PnL
    call_payoff = np.maximum(S_range_1d - K, 0)
    call_pnl = call_payoff - call_price
    
    put_payoff = np.maximum(K - S_range_1d, 0)
    put_pnl = put_payoff - put_price

    fig_pnl = go.Figure()
    # Call
    fig_pnl.add_trace(go.Scatter(x=S_range_1d, y=call_pnl, mode='lines', name='Call PnL', line=dict(color='green')))
    # Put
    fig_pnl.add_trace(go.Scatter(x=S_range_1d, y=put_pnl, mode='lines', name='Put PnL', line=dict(color='red')))
    # Ligne Zéro
    fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
    # Ligne du Spot actuel
    fig_pnl.add_vline(x=S, line_dash="dot", line_color="blue", annotation_text="Current Spot")

    fig_pnl.update_layout(
        xaxis_title="Stock Price at Expiration ($)", 
        yaxis_title="Profit / Loss ($)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

with tab_heatmap:
    st.subheader("Option Value Analysis (Spot vs Volatility)")
    st.caption("Observe how the option premium changes as the stock price and market volatility fluctuate simultaneously.")
    
    # Création des matrices pour la heatmap
    # Axe Y: Prix du sous-jacent (Spot de -20% à +20%)
    spot_range = np.linspace(S * 0.8, S * 1.2, 30)
    # Axe X: Volatilité (de Vol-15% à Vol+15%, borné à 1% min)
    vol_range = np.linspace(max(0.01, sigma - 0.15), sigma + 0.15, 30)
    
    # Meshgrid pour générer la carte 2D
    X, Y = np.meshgrid(vol_range, spot_range)
    
    # Calcul vectorisé (numpy gère ça très bien) pour les heatmaps
    Z_call = np.zeros_like(X)
    Z_put = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z_call[i, j] = black_scholes(Y[i, j], K, T, r, X[i, j], "call")
            Z_put[i, j] = black_scholes(Y[i, j], K, T, r, X[i, j], "put")
            
    col_hm1, col_hm2 = st.columns(2)
    
    with col_hm1:
        fig_call_hm = go.Figure(data=go.Heatmap(
            z=Z_call, x=vol_range, y=spot_range, colorscale='Viridis',
            hovertemplate="Vol: %{x:.1%}<br>Spot: $%{y:.2f}<br>Call Price: $%{z:.2f}<extra></extra>"
        ))
        fig_call_hm.update_layout(title="Call Price Heatmap", xaxis_title="Volatility (σ)", yaxis_title="Stock Price ($)", margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_call_hm, use_container_width=True)

    with col_hm2:
        fig_put_hm = go.Figure(data=go.Heatmap(
            z=Z_put, x=vol_range, y=spot_range, colorscale='Plasma',
            hovertemplate="Vol: %{x:.1%}<br>Spot: $%{y:.2f}<br>Put Price: $%{z:.2f}<extra></extra>"
        ))
        fig_put_hm.update_layout(title="Put Price Heatmap", xaxis_title="Volatility (σ)", yaxis_title="Stock Price ($)", margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_put_hm, use_container_width=True)