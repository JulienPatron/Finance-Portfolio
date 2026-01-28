import streamlit as st

# --- 1. GLOBAL CONFIGURATION ---
st.set_page_config(
    page_title="Julien Patron - Portfolio",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- CSS: SIDEBAR STYLING (CORRECTED) ---
st.markdown("""
<style>
    /* 1. "Julien Patron" Name (Top Left) */
    [data-testid="stSidebarNav"]::before {
        content: "Julien Patron";
        display: block;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
        margin-left: 20px;
        color: var(--text-color);
    }
    
    /* 2. Adjust Spacing */
    div[data-testid="stSidebarNav"] {
        padding-top: 1rem; 
    }

    /* 3. Color Coding for Section Headers (ROBUST METHOD) */
    
    /* Logic: In st.navigation, headers are usually rendered as 'span' elements.
       1st span = " " (Home, hidden)
       2nd span = "Finance"
       3rd span = "Other"
    */

    /* Target the 2nd Header (Finance) -> BLUE */
    div[data-testid="stSidebarNav"] > div > span:nth-of-type(2) {
        color: #1565C0 !important;
        font-weight: 800;
        font-size: 14px;
    }

    /* Target the 3rd Header (Other) -> PURPLE */
    div[data-testid="stSidebarNav"] > div > span:nth-of-type(3) {
        color: #7B1FA2 !important;
        font-weight: 800;
        font-size: 14px;
    }
    
    /* Fallback: In case structure is slightly different (nested divs) */
    div[data-testid="stSidebarNav"] > ul:nth-of-type(2) span { color: #1565C0 !important; }
    div[data-testid="stSidebarNav"] > ul:nth-of-type(3) span { color: #7B1FA2 !important; }

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
        " ": [home_page],              # Header invisible (1st position)
        "Finance": [finance_page],     # Header will be BLUE (2nd position)
        "Other": [movie_page, f1_page] # Header will be PURPLE (3rd position)
    }
)

# --- 4. EXECUTION ---
pg.run()