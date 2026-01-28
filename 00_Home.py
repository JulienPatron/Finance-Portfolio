import streamlit as st

# No set_page_config here, it is handled by main.py

st.title("Welcome to my Portfolio")
st.markdown("### Data Science & Engineering Projects")

st.markdown("""
This application showcases three technical projects demonstrating skills in **Quantitative Finance**, **Machine Learning (NLP)**, and **Data Engineering**.

Please select a project from the sidebar to explore them.
""")

# --- Project Cards ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📈 Finance")
    st.info("**Portfolio Optimizer**")
    st.markdown("""
    Portfolio optimization based on Modern Portfolio Theory (Markowitz).
    * **Tech:** Yahoo Finance, Monte Carlo, Plotly.
    * **Goal:** Maximize the Sharpe Ratio.
    """)

with col2:
    st.subheader("🎬 Cinema")
    st.info("**Movie Recommender**")
    st.markdown("""
    Content-based movie recommendation system (Item-based).
    * **Tech:** Scikit-learn (KNN), TMDB API.
    * **Data:** MovieLens 32M Dataset.
    """)

with col3:
    st.subheader("🏎️ Formula 1")
    st.info("**F1 Elo System**")
    st.markdown("""
    Historical driver ranking based on a custom Elo algorithm.
    * **Tech:** Pandas, Plotly Interactive.
    * **Data:** Comprehensive historical analysis.
    """)

st.divider()
st.caption("Developed by Julien Patron | Hosted on Streamlit Community Cloud")