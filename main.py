import streamlit as st

# --- CONFIGURATION DU ROUTEUR ---
st.set_page_config(layout="wide", page_title="Julien Patron - Portfolio")

# --- DÉFINITION DES PAGES ---

# 1. Portfolio (Il est à la racine, donc on met juste le nom du fichier)
portfolio_page = st.Page(
    "01_Portfolio_Optimizer.py", 
    title="Portfolio Optimizer", 
    icon="📈", 
    default=True
)

# 2. Movie System (Il est dans le dossier pages, donc on met "pages/...")
movie_page = st.Page(
    "pages/03_Movie_Recommendation_System.py", 
    title="Movie Recommendation System", 
    icon="🎬"
)

# --- CRÉATION DE LA NAVIGATION (GROUPÉE) ---
pg = st.navigation(
    {
        "Finance": [portfolio_page],
        "Other(s)": [movie_page],
    }
)

# --- LANCEMENT ---
pg.run()