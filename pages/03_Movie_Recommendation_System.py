import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Movie Recommendation Engine", # Titre de l'onglet du navigateur
    layout="wide"
)

# --- CSS STYLING (Project Standard) ---
st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    h1, h2, h3 {font-family: 'Roboto', sans-serif; font-weight: 400;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. SIDEBAR
# ==============================================================================
st.sidebar.header("User Preferences")
# Placeholder pour les futurs filtres (Genre, Rating, etc.)
st.sidebar.info("Configuration pending...")

# ==============================================================================
# 2. MAIN CONTENT
# ==============================================================================
st.title("Movie Recommendation System")
st.markdown("Content-Based Filtering & Collaborative Filtering Algorithms.")
st.markdown("---")

st.info("🚧 Module under construction. Architecture phase.")