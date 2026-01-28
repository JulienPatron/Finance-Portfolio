import streamlit as st

# Note: No set_page_config here, it is handled by main.py

# --- CSS: TYPOGRAPHY, BADGES & LAYOUT ---
st.markdown("""
<style>
    /* 1. Title Adjustment */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 2. Custom Badge Style for Domains */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 5px;
        color: #333;
    }
    .finance { background-color: #E3F2FD; color: #1565C0; } 
    .cinema  { background-color: #F3E5F5; color: #7B1FA2; } 
    .f1      { background-color: #FFEBEE; color: #C62828; } 
    .data    { background-color: #E8F5E9; color: #2E7D32; } 

    /* 3. Description Text Styling */
    .desc-text {
        font-size: 15px;
        color: #444;
        line-height: 1.5;
    }

    /* 4. EQUAL HEIGHT CARDS HACK (CORRECTED) */
    /* Target the container with border */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        min-height: 280px;      /* Force rigid height */
        height: 100%;
        
        /* Flex Layout Controls */
        display: flex;
        flex-direction: column;
        
        /* ALIGNMENT FIX: Force content to top, not spread out */
        justify-content: flex-start !important; 
        
        /* Spacing between elements (Title, Badge, Text) */
        gap: 15px; 
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("Project Portfolio")

st.markdown("""
This portfolio gathers Python projects applied to market finance and other areas of personal interest.
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
        
        # Domain Badge
        st.markdown('<span class="badge finance">Market Finance</span>', unsafe_allow_html=True)
        
        # Description
        st.markdown("""
        <div class="desc-text">
        Investment tool based on Modern Portfolio Theory. It visualizes the Efficient Frontier to identify the optimal asset allocation for a specific return target.
        </div>
        """, unsafe_allow_html=True)

# 2. CINEMA PROJECT
with col2:
    with st.container(border=True):
        st.page_link("pages/03_Movie_Recommendation_System.py", label="**Movie Recommendation System**", use_container_width=True)
        
        # Domain Badge
        st.markdown('<span class="badge cinema">Cinema & NLP</span>', unsafe_allow_html=True)
        
        # Description
        st.markdown("""
        <div class="desc-text">
        Movie recommendation engine leveraging millions of user ratings to identify fan favorites similar to a selected title.
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
        st.page_link("pages/04_F1_Elo_System.py", label="**F1 Elo Rating System**", use_container_width=True)
        
        # Domain Badge
        st.markdown('<span class="badge f1">Formula 1 Sports Analysis</span>', unsafe_allow_html=True)
        
        # Description
        st.markdown("""
        <div class="desc-text">
        Interactive dashboard applying the Chess Elo rating system to Formula 1 for historical performance analysis.
        </div>
        """, unsafe_allow_html=True)

# 4. FUTURE PROJECT
with col4:
    with st.container(border=True):
        # Button
        st.button("**Upcoming Project**", disabled=True, use_container_width=True)
        
        # Domain Badge
        st.markdown('<span class="badge data">Data Engineering</span>', unsafe_allow_html=True)
        
        # Description
        st.markdown("""
        <div class="desc-text">
        A new infrastructure project is currently under active development. 
        This application will focus on advanced data pipeline automation and cloud engineering concepts, 
        expanding the technical scope of this portfolio.
        </div>
        """, unsafe_allow_html=True)