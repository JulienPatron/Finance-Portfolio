import streamlit as st
import pandas as pd
import pickle
import requests
import os
import gzip

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Project Portfolio - Julien Patron",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    div.stButton > button:first-child {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
    .movie-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA LOADING & API
# ==============================================================================

try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except FileNotFoundError:
    st.error("Error: File .streamlit/secrets.toml not found.")
    st.stop()

@st.cache_resource
def load_engine():
    """Loads model and matrix from compressed files (.gz)."""
    paths = {
        "model": "modele_knn.pkl.gz",     
        "matrix": "matrice_sparse.pkl.gz", 
        "data": "liste_films_final.csv"
    }
    
    if not os.path.exists(paths["data"]):
        paths = {k: f"../{v}" for k, v in paths.items()}

    try:
        with gzip.open(paths["model"], 'rb') as f:
            model = pickle.load(f)
        with gzip.open(paths["matrix"], 'rb') as f:
            matrix = pickle.load(f)
        df = pd.read_csv(paths["data"])
        return model, matrix, df
    except Exception as e:
        st.error(f"Loading error: {e}")
        return None, None, None

def fetch_movie_details(tmdb_id):
    """
    Fetches details (Poster, Overview, Date, Rating) via TMDB in English.
    """
    if pd.isna(tmdb_id):
        return None
    
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        response = requests.get(url, timeout=1.5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# Load Engine
model, matrix, df_movies = load_engine()

if model is None:
    st.stop()

# ==============================================================================
# 3. RABBIT HOLE MANAGEMENT (Session State)
# ==============================================================================

if 'selected_movie_name' not in st.session_state:
    st.session_state['selected_movie_name'] = None

def set_movie(movie_title):
    """Callback function to update session state"""
    st.session_state['selected_movie_name'] = movie_title

# ==============================================================================
# 4. MAIN INTERFACE
# ==============================================================================

st.title("Movie Recommendation System")
st.markdown("From MovieLens 32M Dataset")

# --- INTELLIGENT SEARCH BAR ---
index_to_select = None
if st.session_state['selected_movie_name'] in df_movies['title'].values:
    index_to_select = int(df_movies[df_movies['title'] == st.session_state['selected_movie_name']].index[0])

selected_movie = st.selectbox(
    "Select a reference movie:",
    df_movies['title'].values,
    index=index_to_select,
    placeholder="Type a title (e.g. Inception)...",
)

start_analysis = st.button("Start Analysis", type="primary")

# ==============================================================================
# 5. RECOMMENDATION ENGINE & DISPLAY
# ==============================================================================

# Logic: Display if button clicked OR if a movie is already in memory (Rabbit hole)
if selected_movie and (start_analysis or st.session_state['selected_movie_name']):
    
    # 1. Get Selected Movie Info
    idx = df_movies[df_movies['title'] == selected_movie].index[0]
    tmdb_id_source = df_movies.iloc[idx]['tmdbId']
    matrice_id = df_movies.iloc[idx]['matrice_id']
    
    source_details = fetch_movie_details(tmdb_id_source)
    
    st.divider()
    
    # --- HERO SECTION ---
    col_hero_img, col_hero_txt = st.columns([1, 3])
    
    with col_hero_img:
        if source_details and source_details.get('poster_path'):
            st.image(f"https://image.tmdb.org/t/p/w500{source_details['poster_path']}", use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
            
    with col_hero_txt:
        st.subheader(f"{selected_movie}")
        if source_details:
            date_sortie = source_details.get('release_date', 'Unknown')[:4]
            note = round(source_details.get('vote_average', 0), 1)
            overview = source_details.get('overview', 'No overview available.')
            
            st.caption(f"Year: {date_sortie} | TMDB Rating: {note}/10")
            st.write(f"**Synopsis:** {overview}")
    
    # --- SECTION TITLE ---
    st.write("") 
    st.subheader("Recommended Movies:")
    st.write("") 

    # 2. KNN CALCULATION
    distances, indices = model.kneighbors(matrix[matrice_id], n_neighbors=6)
    
    # 3. RESULTS GRID
    cols = st.columns(5)
    
    for i, col in enumerate(cols):
        neighbor_idx = indices.flatten()[i+1]
        distance = distances.flatten()[i+1]
        similarity = 1 - distance
        
        match = df_movies[df_movies['matrice_id'] == neighbor_idx]
        
        if not match.empty:
            neighbor_data = match.iloc[0]
            neighbor_title = neighbor_data['title']
            neighbor_details = fetch_movie_details(neighbor_data['tmdbId'])
            
            with col:
                # Poster
                if neighbor_details and neighbor_details.get('poster_path'):
                    st.image(f"https://image.tmdb.org/t/p/w500{neighbor_details['poster_path']}", use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
                
                # Title & Year
                year = "????"
                if neighbor_details and neighbor_details.get('release_date'):
                    year = neighbor_details.get('release_date')[:4]
                
                st.markdown(f"**{neighbor_title}** ({year})")
                
                # Similarity Bar
                st.progress(int(similarity * 100))
                st.caption(f"Match: {int(similarity * 100)}%")
                
                # Exploration Button (CORRIGÉ AVEC ON_CLICK)
                st.button(
                    "Search this movie", 
                    key=f"btn_{neighbor_idx}", 
                    on_click=set_movie, 
                    args=(neighbor_title,)
                )

elif not selected_movie:
    st.info("Select a movie from the menu or type a title to start exploring.")