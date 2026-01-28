import streamlit as st

# Note: No set_page_config here, it is handled by main.py

# --- CSS: TYPOGRAPHY, BADGES & LAYOUT ---
st.markdown("""
<style>
    /* 1. Global Spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* 2. Custom Badge Style for Domains */
    .badge {
        display: inline-block;
        padding: 5px 14px;      /* A little more breathing room inside the badge */
        border-radius: 15px;
        font-size: 13px;
        font-weight: 600;
        /* margin-bottom removed, handled by flex gap now */
        color: #333;
        border: 1px solid rgba(0,0,0,0.05); /* Subtle border for definition */
    }
    .finance { background-color: #E3F2FD; color: #1565C0; } 
    .cinema  { background-color: #F3E5F5; color: #7B1FA2; } 
    .f1      { background-color: #FFEBEE; color: #C62828; } 
    .data    { background-color: #E8F5E9; color: #2E7D32; } 

    /* 3. Description Text Styling */
    .desc-text {
        font-size: 15px;
        color: #444;
        line-height: 1.6;       /* Improved line height for readability */
        margin-top: 5px;        /* Slight push from the badge */
    }

    /* 4. EQUAL HEIGHT CARDS HACK & SPACING */
    /* Target the container with border */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        min-height: 300px;      /* Adjusted height */
        height: 100%;
        
        /* Flex Layout Controls */
        display: flex;
        flex-direction: column;
        
        /* ALIGNMENT: Content at top */
        justify-content: flex-start !important; 
        
        /* INTERNAL SPACING: Key change here */
        gap: 22px;              /* Larger gap between Title, Badge, and Text */
        padding-bottom: 15px;   /* Space at the bottom of the card */
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
# Added gap="large" for better horizontal separation
# ROW 1
col1, col2 = st.columns(2, gap="large")

# 1. FINANCE PROJECT
with col1:
    with st.container(border=True):
        # Title
        st.page_link("01_Portfolio_Optimizer.py", label="**Portfolio Optimizer**", use_container_width=True)
        
        # Badge
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
        
        # Badge
        st.markdown('<span class="badge cinema">Cinema & NLP</span>', unsafe_allow_html=True)
        
        # Description
        st.markdown("""
        <div class="desc-text">
        Movie recommendation engine leveraging millions of user ratings to identify fan favorites similar to a selected title.
        </div>
        """, unsafe_allow_html=True)

# Spacer between rows (Increased for better vertical separation)
st.markdown("<br>", unsafe_allow_html=True)

# ROW 2
col3, col4 = st.columns(2, gap="large")

# 3. F1 PROJECT
with col3:
    with st.container(border=True):
        st.page_link("pages/04_F1_Elo_System.py", label="**F1 Elo Rating System**", use_container_width=True)
        
        # Badge
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
        
        # Badge
        st.markdown('<span class="badge data">Data Engineering</span>', unsafe_allow_html=True)
        
        # Description
        st.markdown("""
        <div class="desc-text">
        A new infrastructure project is currently under active development. 
        This application will focus on advanced data pipeline automation and cloud engineering concepts, 
        expanding the technical scope of this portfolio.
        </div>
        """, unsafe_allow_html=True)