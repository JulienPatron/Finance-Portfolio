import streamlit as st

# --- 1. GLOBAL CONFIGURATION ---
st.set_page_config(
    page_title="Julien Patron - Portfolio",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- CSS: SIDEBAR STYLING ---
st.markdown("""
<style>
    /* 1. SIDEBAR BACKGROUND COLOR */
    [data-testid="stSidebar"] {
        background-color: #F8F9FC; /* Very pale blue-grey */
        border-right: 1px solid #E0E0E0; /* Subtle border on the right */
    }

    /* 2. NAME STYLING (Gradient Color) */
    /* Insert text above the navigation container */
    [data-testid="stSidebarNav"]::before {
        content: "Julien Patron";
        display: block;
        font-size: 26px;  
        font-weight: 800; /* Extra Bold */
        margin-bottom: 25px; 
        margin-left: 10px;
        margin-top: 10px;
        
        /* Gradient Text Effect (Blue to Purple) */
        background: linear-gradient(45deg, #1565C0, #7B1FA2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* 3. Adjust Navigation Padding */
    div[data-testid="stSidebarNav"] {
        padding-top: 0.5rem; 
    }
</style>
""", unsafe_allow_html=True)

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
pg = st.navigation(
    {
        " ": [home_page],
        "Finance": [finance_page],
        "Other": [movie_page, f1_page] 
    }
)

# --- 4. EXECUTION ---
pg.run()