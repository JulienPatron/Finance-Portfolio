import streamlit as st

# --- 1. GLOBAL CONFIGURATION ---
st.set_page_config(
    page_title="Julien Patron - Portfolio",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- CSS FOR SIDEBAR STYLING ---
st.markdown("""
<style>
    /* 1. Sidebar Background Color */
    [data-testid="stSidebar"] {
        background-color: #F8F9FB; /* Very light grey-blue */
        border-right: 1px solid #E6E9EF;
    }

    /* 2. "Julien Patron" Name Styling */
    [data-testid="stSidebarNav"]::before {
        content: "Julien Patron";
        display: block;
        font-size: 26px;
        font-weight: 800; /* Extra bold */
        color: #1E3A8A;   /* Royal Blue (Professional) */
        margin-bottom: 25px;
        margin-left: 20px;
        margin-top: 10px;
    }

    /* 3. Navigation Section Headers (Finance, Other) */
    div[data-testid="stSidebarNav"] h4 {
        color: #64748B; /* Slate grey for section titles */
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding-left: 10px;
    }
    
    /* Optional: Style the active link to pop a bit more */
    .st-emotion-cache-16txtl3 {
        color: #1E3A8A !important;
        font-weight: 600;
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