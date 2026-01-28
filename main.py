import streamlit as st

# --- 1. GLOBAL CONFIGURATION ---
st.set_page_config(
    page_title="Julien Patron - Portfolio",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- CSS FOR SIDEBAR NAME & STYLING ---
# This inserts "Julien Patron" above the navigation menu
st.markdown("""
<style>
    /* Insert text above the navigation container */
    [data-testid="stSidebarNav"]::before {
        content: "Julien Patron";
        display: block;
        font-size: 28px;  /* Text size */
        font-weight: bold;
        margin-bottom: 20px; /* Space between name and Home button */
        margin-left: 20px;   /* Alignment */
        color: var(--text-color);
    }
    
    /* Optional: Fine-tune the top padding if needed */
    div[data-testid="stSidebarNav"] {
        padding-top: 1rem; 
    }
</style>
""", unsafe_allow_html=True)

# --- 2. PAGE DEFINITIONS ---

# Landing Page
# MODIFICATION: I removed 'icon="🏠"' here to remove the emoji from the tab and menu
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
    title="Movie Movie Recommendation System"
)

# Project 3: F1
f1_page = st.Page(
    "pages/04_F1_Elo_System.py", 
    title="F1 Elo Rating System"
)

# --- 3. NAVIGATION SETUP ---
# TRICK: using " " for Home hides the section title
pg = st.navigation(
    {
        " ": [home_page],
        "Finance": [finance_page],
        "Other": [movie_page, f1_page] 
    }
)

# --- 4. EXECUTION ---
pg.run()