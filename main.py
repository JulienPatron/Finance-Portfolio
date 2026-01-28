import streamlit as st

# --- 1. GLOBAL CONFIGURATION (Must be the very first command) ---
st.set_page_config(
    page_title="Julien Patron - Portfolio",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- 2. PAGE DEFINITIONS ---
# Icons have been removed as requested.

# Landing Page
home_page = st.Page(
    "00_Home.py", 
    title="Home", 
    default=True
)

# Project 1: Finance
finance_page = st.Page(
    "01_Portfolio_Optimizer.py", 
    title="Portfolio Optimizer"
)

# Project 2: Cinema
movie_page = st.Page(
    "pages/03_Movie_Recommendation_System.py", 
    title="Movie Recommender"
)

# Project 3: F1
f1_page = st.Page(
    "pages/04_F1_Elo_System.py", 
    title="F1 Elo System"
)

# --- 3. NAVIGATION SETUP ---
# Grouped into: Home, Finance, and Other
pg = st.navigation(
    {
        "Home": [home_page],
        "Finance": [finance_page],
        "Other": [movie_page, f1_page],
    }
)

# --- 4. EXECUTION ---
pg.run()