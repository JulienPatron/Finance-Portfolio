import streamlit as st

# --- 1. GLOBAL CONFIGURATION (Must be the very first command) ---
st.set_page_config(
    page_title="Julien Patron - Portfolio",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- 2. PAGE DEFINITIONS ---

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
# TRICK: We use " " (a space) for the Home section key.
# This hides the section label, making "Home" look like a standalone clickable title.
pg = st.navigation(
    {
        " ": [home_page],              # The space hides the header, leaving only the clickable button
        "Finance": [finance_page],     # This creates the "Finance" header
        "Other": [movie_page, f1_page] # This creates the "Other" header
    }
)

# --- 4. EXECUTION ---
pg.run()