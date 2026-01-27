import streamlit as st

# --- CONFIGURATION DU ROUTEUR ---
st.set_page_config(layout="wide", page_title="Julien Patron - Portfolio")

# --- DÉFINITION DES PAGES ---

# 1. Portfolio (Finance)
portfolio_page = st.Page(
    "01_Portfolio_Optimizer.py", 
    title="Portfolio Optimizer",
    default=True
)

# 2. Movie System (Cinema)
movie_page = st.Page(
    "pages/03_Movie_Recommendation_System.py", 
    title="Movie Recommendation System",
)

# 3. F1 Elo System (Sport / Data) - LE NOUVEAU PROJET
f1_page = st.Page(
    "pages/04_F1_Elo_System.py",
    title="F1 Elo Rating System",
)

# --- CRÉATION DE LA NAVIGATION (GROUPÉE) ---
pg = st.navigation(
    {
        "Finance": [portfolio_page],
        "Other Projects": [f1_page, movie_page], # J'ai ajouté la page F1 ici
    }
)

# --- LANCEMENT ---
pg.run()