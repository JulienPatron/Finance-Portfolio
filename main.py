import streamlit as st

# --- 1. CONFIGURATION GLOBALE (Doit être la toute première commande) ---
st.set_page_config(
    page_title="Julien Patron - Portfolio",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- 2. DÉFINITION DES PAGES ---
# On définit les fichiers cibles. 
# Note : title="" définit ce qui apparait dans le menu de navigation.

# Page d'accueil (Load instantané)
home_page = st.Page(
    "00_Home.py", 
    title="Accueil", 
    icon="🏠", 
    default=True
)

# Projet 1 : Finance
finance_page = st.Page(
    "01_Portfolio_Optimizer.py", 
    title="Portfolio Optimizer", 
    icon="📈"
)

# Projet 2 : Cinéma
movie_page = st.Page(
    "pages/03_Movie_Recommendation_System.py", 
    title="Movie Recommender", 
    icon="🎬"
)

# Projet 3 : F1
f1_page = st.Page(
    "pages/04_F1_Elo_System.py", 
    title="F1 Elo System", 
    icon="🏎️"
)

# --- 3. NAVIGATION ---
# Regroupement logique dans la sidebar
pg = st.navigation(
    {
        "Général": [home_page],
        "Projets Data": [finance_page, movie_page, f1_page],
    }
)

# --- 4. EXÉCUTION ---
pg.run()