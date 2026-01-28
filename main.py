import streamlit as st

# --- 1. GLOBAL CONFIGURATION ---
st.set_page_config(
    page_title="Julien Patron - Portfolio",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- CSS FOR SIDEBAR NAME & STYLING ---
st.markdown("""
<style>
    /* Insert text above the navigation container */
    [data-testid="stSidebarNav"]::before {
        content: "Julien Patron";
        display: block;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
        margin-left: 20px;
        color: var(--text-color);
    }
    
    div[data-testid="stSidebarNav"] {
        padding-top: 1rem; 
    }
</style>
""", unsafe_allow_html=True)

# --- 2. PAGE DEFINITIONS ---

home_page = st.Page(
    "00_Home.py", 
    title="Home", 
    default=True
)

# Finance Projects
finance_page_1 = st.Page(
    "01_Portfolio_Optimizer.py", 
    title="Portfolio Optimizer"
)

finance_page_2 = st.Page(
    "pages/05_Black_Scholes_Option_Pricer.py", 
    title="Black-Scholes Pricer"
)

# Other Projects
movie_page = st.Page(
    "pages/03_Movie_Recommendation_System.py", 
    title="Movie Recommender"
)

f1_page = st.Page(
    "pages/04_F1_Elo_System.py", 
    title="F1 Elo System"
)

# --- 3. NAVIGATION SETUP ---
pg = st.navigation(
    {
        " ": [home_page],
        "Finance": [finance_page_1, finance_page_2], # Ajout ici
        "Other": [movie_page, f1_page] 
    }
)

# --- 4. EXECUTION ---
pg.run()