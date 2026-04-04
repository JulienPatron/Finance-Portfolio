import streamlit as st
import pandas as pd
import pickle
import requests
import os
import gzip
import gc

st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    .movie-title {
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
    }
    .movie-meta {
        text-align: center; 
        margin-top: -10px; 
        font-size: 14px; 
        color: #555;
    }
    .provider-logo {
        width: 35px; 
        margin: 0 4px; 
        border-radius: 5px;
    }
    .provider-container {
        height: 45px; 
        margin-top: 12px; 
        margin-bottom: 12px; 
        display: flex; 
        align-items: center; 
        justify-content: center;
    }
    .match-text {
        text-align: center;
        font-size: 14px;
        margin-bottom: 5px;
        color: #000000;
    }
    .poster-container {
        position: relative;
        width: 100%;
        aspect-ratio: 2 / 3;
        margin-bottom: 10px;
        border-radius: 5px;
        overflow: hidden;
    }
    .movie-poster {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_recommendation_engine():
    try:
        file_paths = {
            "model": "modele_knn.pkl.gz",
            "matrix": "matrice_sparse.pkl.gz",
            "db": "liste_films_final.csv"
        }
        
        if not os.path.exists(file_paths["db"]):
             file_paths = {k: f"../{v}" for k, v in file_paths.items()}

        with gzip.open(file_paths["model"], 'rb') as f:
            model = pickle.load(f)
        gc.collect() 

        with gzip.open(file_paths["matrix"], 'rb') as f:
            matrix = pickle.load(f)
        gc.collect()

        df = pd.read_csv(file_paths["db"])
        
        return model, matrix, df
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

model, matrix, df = load_recommendation_engine()

if model is None:
    st.warning("Model files (pkl.gz) not found.")
    st.stop()

@st.cache_data(ttl=86400, show_spinner=False) 
def get_tmdb_details(tmdb_id):
    if pd.isna(tmdb_id): 
        return None
    
    try:
        api_key = st.secrets["TMDB_API_KEY"]
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}&language=en-US&append_to_response=watch/providers"
        
        response = requests.get(url, timeout=1.5) 
        if response.status_code != 200:
            return None
            
        data = response.json()
        
        providers = data.get('watch/providers', {}).get('results', {}).get('FR', {}).get('flatrate', [])[:3]
        logos = [{
            "logo": f"https://image.tmdb.org/t/p/original{p['logo_path']}",
            "name": p['provider_name']
        } for p in providers]

        return {
            "poster": f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}" if data.get('poster_path') else "https://via.placeholder.com/300x450?text=No+Image",
            "title": data.get('title'),
            "overview": data.get('overview', 'Synopsis unavailable.'),
            "year": data.get('release_date', '????')[:4],
            "rating": round(data.get('vote_average', 0), 1),
            "genres": ", ".join([g['name'] for g in data.get('genres', [])][:3]),
            "runtime": f"{data.get('runtime', 0)//60}h {data.get('runtime', 0)%60:02d}m",
            "streaming": logos
        }
    except:
        return None

if 'movie' not in st.session_state:
    st.session_state['movie'] = None

def update_selection(title):
    st.session_state['movie'] = title

st.title("Movie Recommendation System")
st.markdown(" Movie recommendation engine using millions of user ratings to identify fan favorites based on a selected title.")

idx = int(df[df['title'] == st.session_state['movie']].index[0]) if st.session_state['movie'] in df['title'].values else None
selected = st.selectbox(
    "Search for a reference movie released before June 2023:", 
    df['title'].values, 
    index=idx, 
    placeholder="Type a title (e.g. Inception)..."
)

go_btn = st.button("Get Recommendations", type="primary", use_container_width=True)

if selected and (go_btn or st.session_state['movie']):
    
    row = df[df['title'] == selected].iloc[0]
    info = get_tmdb_details(row['tmdbId'])
    
    if info is None:
        info = {"poster": "https://via.placeholder.com/300x450?text=Error", "year": "????", 
                "runtime": "N/A", "genres": "N/A", "rating": "N/A", 
                "overview": "Information unavailable (API Error).", "streaming": []}

    st.divider()
    
    c1, c2 = st.columns([1, 3])
    with c1:
        st.image(info['poster'], use_container_width=True)
    with c2:
        st.subheader(selected)
        st.caption(f"Year: {info['year']} | Runtime: {info['runtime']} | Genres: {info['genres']}")
        st.markdown(f"**TMDB Rating:** {info['rating']}/10")
        st.write(f"_{info['overview']}_")
        
        if info['streaming']:
            st.markdown("**Available on (FR):**")
            logos_html = "".join([f'<img src="{p["logo"]}" class="provider-logo" title="{p["name"]}">' for p in info['streaming']])
            st.markdown(logos_html, unsafe_allow_html=True)

    st.subheader("Fans also liked:")
    
    distances, indices = model.kneighbors(matrix[row['matrice_id']], n_neighbors=6)
    
    cols = st.columns(5)
    
    for i, col in enumerate(cols):
        neighbor_idx = i + 1
        neighbor_db_id = indices.flatten()[neighbor_idx]
        neighbor_dist = distances.flatten()[neighbor_idx]
        
        neighbor_row = df[df['matrice_id'] == neighbor_db_id].iloc[0]
        neighbor_title = neighbor_row['title']
        
        n_info = get_tmdb_details(neighbor_row['tmdbId'])
        
        if n_info is None:
            n_info = {"poster": "https://via.placeholder.com/300x450?text=Unavailable", "rating": "N/A", "streaming": []}
        
        match_score = int((1 - neighbor_dist) * 100)

        with col:
            st.markdown(f'<div class="poster-container"><img src="{n_info["poster"]}" class="movie-poster"></div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="movie-title">{neighbor_title}</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="match-text">Match: {match_score}%</div>', unsafe_allow_html=True)
            st.progress(match_score)
            
            st.markdown(f'<div class="movie-meta">Rating: {n_info["rating"]}</div>', unsafe_allow_html=True)

            if n_info['streaming']:
                logos_html = "".join([f'<img src="{p["logo"]}" class="provider-logo" title="{p["name"]}">' for p in n_info['streaming']])
                st.markdown(f'<div class="provider-container">{logos_html}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="provider-container" style="color:#ccc; font-size:12px;">Not available</div>', unsafe_allow_html=True)

            st.button(
                "Select", 
                key=f"btn_{neighbor_db_id}", 
                on_click=update_selection, 
                args=(neighbor_title,), 
                use_container_width=True
            )