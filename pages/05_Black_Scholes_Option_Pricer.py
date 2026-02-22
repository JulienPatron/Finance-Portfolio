import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as si
from scipy.interpolate import griddata
import plotly.graph_objects as go
import yfinance as yf
import datetime

# --- CONFIGURATION DE LA PAGE ---
# st.set_page_config(page_title="Black-Scholes Pricer", layout="wide") 

# ==============================================================================
# 1. MOTEUR MATHÉMATIQUE (BLACK-SCHOLES)
# ==============================================================================

def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    
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
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        if hist.empty: return None
        current_price = float(hist['Close'].iloc[-1])
        
        hist['Log_Ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
        volatility = float(hist['Log_Ret'].std() * np.sqrt(252))
        
        tnx = yf.Ticker("^TNX")
        tnx_hist = tnx.history(period="5d")
        rf_rate = float(tnx_hist['Close'].iloc[-1]) / 100.0 if not tnx_hist.empty else 0.04
            
        return {"price": current_price, "vol": volatility, "rf_rate": rf_rate}
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_iv_surface(ticker_symbol):
    """Récupère les chaînes d'options réelles pour construire la surface de volatilité."""
    try:
        stock = yf.Ticker(ticker_symbol)
        expirations = stock.options
        if not expirations: return None
        
        # On limite aux 6 prochaines expirations pour ne pas bloquer l'API
        expirations = expirations[:6]
        
        iv_data = []
        today = datetime.date.today()
        
        for date in expirations:
            exp_date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
            T = (exp_date - today).days / 365.0
            if T <= 0.01: continue # On ignore les options qui expirent aujourd'hui
            
            chain = stock.option_chain(date)
            calls = chain.calls
            
            # On filtre pour garder des données cohérentes (IV > 0 et un peu de volume)
            calls = calls[(calls['impliedVolatility'] > 0.01) & (calls['volume'] > 0)]
            
            for _, row in calls.iterrows():
                iv_data.append([row['strike'], T, row['impliedVolatility']])
                
        if not iv_data: return None
        return pd.DataFrame(iv_data, columns=['Strike', 'Maturity', 'IV'])
    except Exception:
        return None

# ==============================================================================
# 3. GESTION DES VARIABLES D'ÉTAT (SESSION STATE)
# ==============================================================================

if 'S' not in st.session_state: st.session_state['S'] = 100.0
if 'K' not in st.session_state: st.session_state['K'] = 100.0
if 'vol' not in st.session_state: st.session_state['vol'] = 0.20
if 'rf' not in st.session_state: st.session_state['rf'] = 0.04
if 'ticker' not in st.session_state: st.session_state['ticker'] = "AAPL"

# ==============================================================================
# 4. INTERFACE UTILISATEUR (SIDEBAR & INPUTS)
# ==============================================================================

st.title("📈 Black-Scholes Option Pricing & Analytics")

with st.sidebar:
    st.header("1. Market Data (API)")
    ticker = st.text_input("Ticker Yahoo Finance (ex: AAPL, TSLA)", st.session_state['ticker'])
    
    if st.button("Fetch Live Data", type="primary"):
        with st.spinner("Fetching data..."):
            data = fetch_market_data(ticker)
            if data:
                st.session_state['ticker'] = ticker
                st.session_state['S'] = data['price']
                st.session_state['K'] = round(data['price'], 2)
                st.session_state['vol'] = data['vol']
                st.session_state['rf'] = data['rf_rate']
                st.success("Data updated!")
            else:
                st.error("Ticker not found.")
                
    st.divider()
    
    st.header("2. Manual Overrides")
    st.caption("Ajustez ces valeurs pour simuler vos propres scénarios.")
    
    S = st.number_input("Stock Price ($)", min_value=1.0, value=float(st.session_state['S']), step=1.0)
    K = st.number_input("Strike Price ($)", min_value=1.0, value=float(st.session_state['K']), step=1.0)
    
    days_to_maturity = st.slider("Days to Maturity", min_value=1, max_value=365, value=30)
    T = days_to_maturity / 365.0 
    
    sigma = st.slider("Volatility (σ)", min_value=0.01, max_value=1.50, value=float(st.session_state['vol']), step=0.01)
    r = st.slider("Risk-Free Rate (r)", min_value=0.00, max_value=0.15, value=float(st.session_state['rf']), step=0.01)

# ==============================================================================
# 5. CALCULS EN DIRECT & KPI DASHBOARD
# ==============================================================================

call_price = black_scholes(S, K, T, r, sigma, "call")
put_price = black_scholes(S, K, T, r, sigma, "put")

st.subheader("Pricing Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Underlying Price", f"${S:,.2f}")
col2.metric("Call Premium", f"${call_price:,.2f}")
col3.metric("Put Premium", f"${put_price:,.2f}")
col4.metric("Time to Expiry", f"{days_to_maturity} Days")

st.divider()

# ==============================================================================
# 6. GRAPHIQUES (PAYOFF, HEATMAPS & IV SURFACE)
# ==============================================================================

tab_payoff, tab_heatmap, tab_surface = st.tabs(["PnL & Payoff", "Pricing Heatmaps", "Real IV Surface (3D)"])

# --- TAB 1: PAYOFF ---
with tab_payoff:
    st.subheader("Profit & Loss at Expiration")
    S_range_1d = np.linspace(S * 0.7, S * 1.3, 100)
    
    call_payoff = np.maximum(S_range_1d - K, 0)
    call_pnl = call_payoff - call_price
    
    put_payoff = np.maximum(K - S_range_1d, 0)
    put_pnl = put_payoff - put_price

    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(x=S_range_1d, y=call_pnl, mode='lines', name='Call PnL', line=dict(color='green')))
    fig_pnl.add_trace(go.Scatter(x=S_range_1d, y=put_pnl, mode='lines', name='Put PnL', line=dict(color='red')))
    fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_pnl.add_vline(x=S, line_dash="dot", line_color="blue", annotation_text="Current Spot")

    fig_pnl.update_layout(xaxis_title="Stock Price at Expiration ($)", yaxis_title="Profit / Loss ($)", hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_pnl, use_container_width=True)

# --- TAB 2: HEATMAPS ---
with tab_heatmap:
    st.subheader("Option Value Analysis (Spot vs Volatility)")
    
    spot_range = np.linspace(S * 0.8, S * 1.2, 30)
    vol_range = np.linspace(max(0.01, sigma - 0.15), sigma + 0.15, 30)
    
    X, Y = np.meshgrid(vol_range, spot_range)
    Z_call = np.zeros_like(X)
    Z_put = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z_call[i, j] = black_scholes(Y[i, j], K, T, r, X[i, j], "call")
            Z_put[i, j] = black_scholes(Y[i, j], K, T, r, X[i, j], "put")
            
    col_hm1, col_hm2 = st.columns(2)
    
    with col_hm1:
        fig_call_hm = go.Figure(data=go.Heatmap(z=Z_call, x=vol_range, y=spot_range, colorscale='Viridis'))
        fig_call_hm.update_layout(title="Call Price", xaxis_title="Volatility (σ)", yaxis_title="Stock Price ($)", margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_call_hm, use_container_width=True)

    with col_hm2:
        fig_put_hm = go.Figure(data=go.Heatmap(z=Z_put, x=vol_range, y=spot_range, colorscale='Plasma'))
        fig_put_hm.update_layout(title="Put Price", xaxis_title="Volatility (σ)", yaxis_title="Stock Price ($)", margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_put_hm, use_container_width=True)

# --- TAB 3: 3D IV SURFACE ---
with tab_surface:
    st.subheader(f"Real Implied Volatility Surface: {st.session_state['ticker']}")
    st.caption("Données récupérées en direct sur le marché des options via Yahoo Finance.")
    
    with st.spinner("Construction de la surface 3D..."):
        df_iv = fetch_iv_surface(st.session_state['ticker'])
        
        if df_iv is not None and not df_iv.empty:
            # On crée une grille 2D régulière pour la 3D (Interpolation)
            strikes = np.linspace(df_iv['Strike'].min(), df_iv['Strike'].max(), 50)
            maturities = np.linspace(df_iv['Maturity'].min(), df_iv['Maturity'].max(), 50)
            grid_strikes, grid_maturities = np.meshgrid(strikes, maturities)
            
            # Interpolation des points manquants
            grid_iv = griddata(
                (df_iv['Strike'], df_iv['Maturity']), 
                df_iv['IV'], 
                (grid_strikes, grid_maturities), 
                method='linear'
            )
            
            fig_surf = go.Figure(data=[go.Surface(
                z=grid_iv, x=grid_strikes, y=grid_maturities, colorscale='Inferno',
                colorbar=dict(title='Implied Volatility')
            )])
            
            fig_surf.update_layout(
                title=f"IV Surface ({st.session_state['ticker']})",
                scene=dict(
                    xaxis_title='Strike Price ($)',
                    yaxis_title='Time to Maturity (Years)',
                    zaxis_title='Implied Volatility (IV)'
                ),
                height=600,
                margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig_surf, use_container_width=True)
        else:
            st.warning("Impossible de récupérer la chaîne d'options pour ce Ticker. (Vérifiez s'il possède bien des options négociables).")