import streamlit as st

# Note: No set_page_config here, it is handled by main.py

# --- CSS: ADJUST LAYOUT & TYPOGRAPHY ---
st.markdown("""
<style>
    /* 1. Reduce top padding to move title higher */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
    
    /* 2. Custom style for the Title Links to look like Headers */
    /* Only applied to button-like elements inside columns if needed, 
       but standard page_link is sufficient with bold text. */
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("Project Portfolio")

# Updated intro: No bold on "market finance", simpler text
st.markdown("""
This portfolio gathers Python projects applied to market finance and other areas of personal interest.
""")

st.divider()

# --- PROJECT SLOTS (4 Columns) ---
col1, col2, col3, col4 = st.columns(4)

# 1. FINANCE PROJECT
with col1:
    # The Title IS the Link (Big & Clickable)
    st.page_link("01_Portfolio_Optimizer.py", label="**Portfolio Optimizer**", use_container_width=True)
    
    # Domain (Small & Below)
    st.caption("Finance")
    
    # Description (Short)
    st.markdown("Strategic allocation tool based on Markowitz theory.")

# 2. CINEMA PROJECT
with col2:
    st.page_link("pages/03_Movie_Recommendation_System.py", label="**Movie Recommender**", use_container_width=True)
    st.caption("Cinema")
    st.markdown("Suggestions based on content similarity.")

# 3. F1 PROJECT
with col3:
    st.page_link("pages/04_F1_Elo_System.py", label="**F1 Elo System**", use_container_width=True)
    st.caption("Formula 1")
    st.markdown("Historical driver ranking using Elo algorithm.")

# 4. FUTURE PROJECT
with col4:
    # Placeholder button (Disabled)
    st.button("**Upcoming Project**", disabled=True, use_container_width=True)
    st.caption("Data Engineering")
    st.markdown("Currently under development.")