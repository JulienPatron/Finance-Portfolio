import streamlit as st

# Note: No set_page_config here, it is handled by main.py

# --- TITLE & INTRO ---
st.title("Project Portfolio")

st.markdown("""
This portfolio gathers Python projects applied to **Market Finance** and other areas of personal interest. 
It demonstrates the ability to build interactive tools that solve specific analytical problems.
""")

st.markdown("---")

# --- PROJECT SLOTS (4 Columns) ---
col1, col2, col3, col4 = st.columns(4)

# 1. FINANCE PROJECT
with col1:
    st.subheader("Finance")
    st.markdown("**Portfolio Optimizer**")
    st.write("""
    A strategic investment tool designed to construct efficient portfolios. 
    It balances risk and return using historical market data to suggest optimal asset allocations.
    """)
    st.page_link("01_Portfolio_Optimizer.py", label="Open Project", icon=None, use_container_width=True)

# 2. CINEMA PROJECT
with col2:
    st.subheader("Cinema")
    st.markdown("**Movie Recommender**")
    st.write("""
    An intelligent discovery engine that suggests movies based on your preferences. 
    It analyzes similarity patterns between films to provide personalized recommendations.
    """)
    st.page_link("pages/03_Movie_Recommendation_System.py", label="Open Project", icon=None, use_container_width=True)

# 3. F1 PROJECT
with col3:
    st.subheader("Formula 1")
    st.markdown("**F1 Elo System**")
    st.write("""
    A historical ranking system for Formula 1 drivers. 
    It uses a comparative algorithm to objectively measure driver performance and dominance across different eras.
    """)
    st.page_link("pages/04_F1_Elo_System.py", label="Open Project", icon=None, use_container_width=True)

# 4. FUTURE PROJECT (Placeholder)
with col4:
    st.subheader("Upcoming")
    st.markdown("**Future Project**")
    st.write("""
    A new project is currently under development. 
    This slot is reserved for a future application expanding on data analysis capabilities.
    """)
    # Disabled button for the placeholder
    st.button("Coming Soon", disabled=True, use_container_width=True)