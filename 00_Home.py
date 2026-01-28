import streamlit as st

# Note: No set_page_config here, it is handled by main.py

# --- CSS: TYPOGRAPHY & BADGES ---
st.markdown("""
<style>
    /* 1. Title Adjustment */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 2. Custom Badge Style for Domains (Pale Colors) */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 10px;
        color: #333;
    }
    .finance { background-color: #E3F2FD; color: #1565C0; } /* Pale Blue */
    .cinema  { background-color: #F3E5F5; color: #7B1FA2; } /* Pale Purple */
    .f1      { background-color: #FFEBEE; color: #C62828; } /* Pale Red */
    .data    { background-color: #E8F5E9; color: #2E7D32; } /* Pale Green */

    /* 3. Description Text Styling */
    .desc-text {
        font-size: 15px;
        color: #444;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("Project Portfolio")

st.markdown("""
This portfolio gathers Python projects applied to market finance and other areas of personal interest. 
Below you will find interactive tools designed to solve specific analytical challenges.
""")

st.divider()

# --- PROJECT GRID (2x2 Layout) ---

# ROW 1
col1, col2 = st.columns(2, gap="medium")

# 1. FINANCE PROJECT
with col1:
    with st.container(border=True):
        # Title (Clickable)
        st.page_link("01_Portfolio_Optimizer.py", label="**Portfolio Optimizer**", use_container_width=True)
        
        # Domain Badge (Pale Blue)
        st.markdown('<span class="badge finance">Market Finance</span>', unsafe_allow_html=True)
        
        # Expanded Description
        st.markdown("""
        <div class="desc-text">
        A comprehensive investment tool based on Modern Portfolio Theory (Markowitz). 
        It processes historical market data to generate the Efficient Frontier, allowing users to visualize risk-return trade-offs 
        and determine the optimal asset allocation for their specific financial goals.
        </div>
        """, unsafe_allow_html=True)

# 2. CINEMA PROJECT
with col2:
    with st.container(border=True):
        st.page_link("pages/03_Movie_Recommendation_System.py", label="**Movie Recommender**", use_container_width=True)
        
        # Domain Badge (Pale Purple)
        st.markdown('<span class="badge cinema">Cinema & NLP</span>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="desc-text">
        An intelligent discovery engine leveraging content-based filtering techniques. 
        By analyzing metadata and textual descriptions from the MovieLens 32M dataset, 
        this system identifies similarity patterns to suggest movies that align with your unique viewing preferences.
        </div>
        """, unsafe_allow_html=True)

# Spacer between rows
st.write("") 
st.write("")

# ROW 2
col3, col4 = st.columns(2, gap="medium")

# 3. F1 PROJECT
with col3:
    with st.container(border=True):
        st.page_link("pages/04_F1_Elo_System.py", label="**F1 Elo System**", use_container_width=True)
        
        # Domain Badge (Pale Red)
        st.markdown('<span class="badge f1">Formula 1 Sports Analysis</span>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="desc-text">
        A robust historical ranking system for Formula 1 drivers using a custom Elo algorithm. 
        Unlike standard points systems, this model accounts for teammate comparisons and field strength to objectively measure 
        driver dominance across different eras of the sport.
        </div>
        """, unsafe_allow_html=True)

# 4. FUTURE PROJECT
with col4:
    with st.container(border=True):
        # Button is disabled but styled similarly
        st.button("**Upcoming Project**", disabled=True, use_container_width=True)
        
        # Domain Badge (Pale Green)
        st.markdown('<span class="badge data">Data Engineering</span>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="desc-text">
        A new infrastructure project is currently under active development. 
        This application will focus on advanced data pipeline automation and cloud engineering concepts, 
        expanding the technical scope of this portfolio.
        </div>
        """, unsafe_allow_html=True)