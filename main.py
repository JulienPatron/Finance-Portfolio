import streamlit as st

st.set_page_config(
    page_title="Julien Patron - Portfolio",
    layout="wide",
    initial_sidebar_state="expanded" 
)

st.markdown("""
<style>
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

if st.query_params.get("admin") == "true":
    st.session_state["admin_mode"] = True

home_page = st.Page("00_Home.py", title="Home", default=True)
finance_page_1 = st.Page("01_Portfolio_Optimizer.py", title="Portfolio Optimizer")
finance_page_2 = st.Page("pages/02_Equity_Valuation_Model.py", title="Equity Valuation Model (CAPM)")
movie_page = st.Page("pages/03_Movie_Recommendation_System.py", title="Movie Recommendation System")
f1_page = st.Page("pages/04_F1_Elo_System.py", title="F1 Elo Rating System")

nav_dict = {
    " ": [home_page],
    "Finance": [finance_page_1, finance_page_2],
    "Other": [movie_page, f1_page] 
}

if st.session_state.get("admin_mode", False):
    nav_dict["Private"] = [st.Page("pages/99_Watchlist.py", title="Watchlist", url_path="watchlist")]

pg = st.navigation(nav_dict)
pg.run()