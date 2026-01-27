import streamlit as st

# Pas de set_page_config ici, il est géré par main.py

st.title("Bienvenue sur mon Portfolio")
st.markdown("### Data Science & Engineering Projects")

st.markdown("""
Cette application regroupe trois projets techniques démontrant des compétences en **Finance Quantitative**, **Machine Learning (NLP)** et **Data Engineering**.

Veuillez sélectionner un projet dans la barre latérale pour commencer.
""")

# --- Présentation rapide des projets (Cartes) ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📈 Finance")
    st.info("**Portfolio Optimizer**")
    st.markdown("""
    Optimisation de portefeuille basée sur la théorie moderne (Markowitz).
    * **Tech:** Yahoo Finance, Monte Carlo, Plotly.
    * **Objectif:** Maximiser le ratio de Sharpe.
    """)

with col2:
    st.subheader("🎬 Cinéma")
    st.info("**Movie Recommender**")
    st.markdown("""
    Système de recommandation de films basé sur le contenu (Item-based).
    * **Tech:** Scikit-learn (KNN), TMDB API.
    * **Data:** MovieLens 32M Dataset.
    """)

with col3:
    st.subheader("🏎️ Formule 1")
    st.info("**F1 Elo System**")
    st.markdown("""
    Classement historique des pilotes basé sur un algorithme Elo personnalisé.
    * **Tech:** Pandas, Plotly Interactive.
    * **Data:** Analyse historique complète.
    """)

st.divider()
st.caption("Développé par Julien Patron | Hébergé sur Streamlit Community Cloud")