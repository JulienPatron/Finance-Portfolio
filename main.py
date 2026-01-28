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
    /* 1. "Julien Patron" Name (Top Left) - Keep Dark/Neutral */
    [data-testid="stSidebarNav"]::before {
        content: "Julien Patron";
        display: block;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
        margin-left: 20px;
        color: var(--text-color); /* Neutral Dark */
    }
    
    /* 2. Adjust Spacing */
    div[data-testid="stSidebarNav"] {
        padding-top: 1rem; 
    }

    /* 3. Color Coding for Section Headers */
    
    /* Target the Section Headers within the navigation */
    /* Note: Streamlit renders section headers as span elements inside a specific div structure. 
       We target them by order (nth-of-type) to apply specific colors. */

    /* "Finance" Header (2nd Section, assuming " " is 1st) -> BLUE */
    div[data-testid="stSidebarNav"] > ul:nth-of-type(2) span {
        color: #1565C0 !important; /* Matches Home Finance Badge */
        font-weight: 800;
        font-size: 14px;
    }

    /* "Other" Header (3rd Section) -> PURPLE */
    div[data-testid="stSidebarNav"] > ul:nth-of-type(3) span {
        color: #7B1FA2 !important; /* Matches Home Cinema Badge */
        font-weight: 800;
        font-size: 14px;
    }
    
    /* Optional: Style the active page background to be subtle gray */
    .st-emotion-cache-16txtl3 {
        background-color: #f0f2f6;
    }

</style>
""", unsafe_allow_html=True)

# --- 2. PAGE DEFINITIONS ---

home_page = st.Page(
    "00_Home.py", 
    title="Home", 
    default=True
)

finance_page = st.Page(
    "01_Portfolio_Optimizer.py", 
    title="Portfolio Optimizer"
)

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
        " ": [home_page],              # Header invisible
        "Finance": [finance_page],     # Header will be BLUE
        "Other": [movie_page, f1_page] # Header will be PURPLE
    }
)

# --- 4. EXECUTION ---
pg.run()