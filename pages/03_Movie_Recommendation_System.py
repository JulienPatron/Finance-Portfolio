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
    Fetches details + Watch Providers via TMDB (Version FR).
    """
    if pd.isna(tmdb_id):
        return None
    
    # On garde l'anglais pour le texte (synopsis), mais on veut les infos de streaming...
    # Note: L'API renvoie les providers liés à la région demandée plus bas, pas la langue.
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}&language=en-US&append_to_response=watch/providers"
    
    try:
        response = requests.get(url, timeout=1.5)
        if response.status_code == 200:
            data = response.json()
            
            # --- 1. TRAITEMENT CLASSIQUE ---
            genres_list = [g['name'] for g in data.get('genres', [])]
            genres_str = ", ".join(genres_list[:3])
            
            runtime_min = data.get('runtime', 0)
            runtime_str = f"{runtime_min // 60}h {runtime_min % 60:02d}m" if runtime_min else "N/A"

            # --- 2. TRAITEMENT STREAMING (CORRECTION FR 🇫🇷) ---
            # On change 'US' par 'FR' ici !
            providers = data.get('watch/providers', {}).get('results', {}).get('FR', {})
            
            # On cherche les plateformes de streaming "Flatrate" (Abonnement)
            flatrate = providers.get('flatrate', [])
            
            streaming_list = []
            for p in flatrate[:3]:
                streaming_list.append({
                    "name": p['provider_name'],
                    "logo": f"https://image.tmdb.org/t/p/original{p['logo_path']}"
                })

            return {
                "poster_path": data.get("poster_path"),
                "backdrop_path": data.get("backdrop_path"),
                "title": data.get("title"),
                "overview": data.get("overview", "No overview available."),
                "release_year": data.get("release_date", "Unknown")[:4],
                "rating": round(data.get("vote_average", 0), 1),
                "genres": genres_str,
                "runtime": runtime_str,
                "streaming": streaming_list
            }
    except Exception as e:
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

# La phrase dynamique demandée :
st.markdown("Item-based filtering using user ratings from the MovieLens 32M dataset | Movie data from TMDB | Period: 1902 - 2023")

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
            # Metadata Line
            st.caption(f"Year: {source_details['release_year']} | Runtime: {source_details['runtime']} | Genres: {source_details['genres']}")
            
            # Rating
            st.write(f"**TMDB Rating:** {source_details['rating']}/10")
            
            # Synopsis
            st.write(f"**Synopsis:** {source_details['overview']}")
            
            # Streaming Availability (Hero Section)
            if source_details.get('streaming'):
                st.write("")
                st.markdown("**Available on:**")
                logos_html = ""
                for p in source_details['streaming']:
                    logos_html += f'<img src="{p["logo"]}" style="width:60px; margin-right:10px; border-radius:8px;" title="{p["name"]}">'
                st.markdown(logos_html, unsafe_allow_html=True)

    # --- SECTION TITLE ---
    st.write("") 
    st.subheader("Recommended Movies:")
    st.write("") 

    # 2. KNN CALCULATION
    distances, indices = model.kneighbors(matrix[matrice_id], n_neighbors=6)
    
    # 3. RESULTS GRID
    cols = st.columns(5)
    
    for i, col in enumerate(cols):
        neighbor_idx = indices.flatten()[i+1] # Skip self
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
                if neighbor_details and neighbor_details.get('release_year'):
                    year = neighbor_details.get('release_year')
                
                # Titre & Année (Correction Hauteur Fixe)
                # On crée un bloc HTML de 50px de haut pour forcer l'alignement
                title_html = f"""
                <div style="
                    height: 50px; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    text-align: center; 
                    font-weight: bold; 
                    font-size: 14px;
                    margin-bottom: 5px;
                    line-height: 1.2;
                    overflow: hidden;
                    text-overflow: ellipsis;
                ">
                    {neighbor_title} ({year})
                </div>
                """
                st.markdown(title_html, unsafe_allow_html=True)
                
                # Similarity Bar
                st.progress(int(similarity * 100))
                st.caption(f"Match: {int(similarity * 100)}%")
                
                # Streaming Logos (Small versions for cards)
                if neighbor_details and neighbor_details.get('streaming'):
                    logos_html = ""
                    for p in neighbor_details['streaming']:
                        logos_html += f'<img src="{p["logo"]}" style="width:35px; margin-right:5px; border-radius:5px;" title="{p["name"]}">'
                    st.markdown(logos_html, unsafe_allow_html=True)
                    st.write("") # Spacer

                # Exploration Button (Fixed with Callback)
                st.button(
                    "Search this movie", 
                    key=f"btn_{neighbor_idx}", 
                    on_click=set_movie, 
                    args=(neighbor_title,)
                )

elif not selected_movie:
    st.info("Select a movie from the menu or type a title to start exploring.")