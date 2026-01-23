import streamlit as st
import pandas as pd
import pickle, requests, os, gzip

# 1. CONFIGURATION & STYLE
st.set_page_config(page_title="Project Portfolio", layout="wide")
st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    /* On garde le style pour l'arrondi et le gras, mais la largeur sera gérée par Python */
    div.stButton > button:first-child {
        border-radius: 8px; 
        font-weight: bold; 
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# 2. CHARGEMENT DES DONNÉES
@st.cache_resource
def load_data():
    try:
        files = {"model": "modele_knn.pkl.gz", "matrix": "matrice_sparse.pkl.gz", "db": "liste_films_final.csv"}
        if not os.path.exists(files["db"]): 
            files = {k: f"../{v}" for k, v in files.items()}

        with gzip.open(files["model"], 'rb') as f: model = pickle.load(f)
        with gzip.open(files["matrix"], 'rb') as f: matrix = pickle.load(f)
        return model, matrix, pd.read_csv(files["db"])
    except: return None, None, None

model, matrix, df = load_data()
if model is None: st.stop()

# 3. FONCTION API TMDB
def get_details(tmdb_id):
    if pd.isna(tmdb_id): return None
    api_key = st.secrets["TMDB_API_KEY"]
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}&language=en-US&append_to_response=watch/providers"
    
    try:
        data = requests.get(url, timeout=1).json()
        providers = data.get('watch/providers', {}).get('results', {}).get('FR', {}).get('flatrate', [])[:3]
        logos = [{"logo": f"https://image.tmdb.org/t/p/original{p['logo_path']}", "name": p['provider_name']} for p in providers]

        return {
            "poster": f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}" if data.get('poster_path') else "https://via.placeholder.com/300x450?text=No+Image",
            "title": data.get('title'),
            "overview": data.get('overview', 'No synopsis.'),
            "year": data.get('release_date', '????')[:4],
            "rating": round(data.get('vote_average', 0), 1),
            "genres": ", ".join([g['name'] for g in data.get('genres', [])][:3]),
            "runtime": f"{data.get('runtime', 0)//60}h {data.get('runtime', 0)%60:02d}m",
            "streaming": logos
        }
    except: return None

# 4. INTERFACE & NAVIGATION
if 'movie' not in st.session_state: st.session_state['movie'] = None

def update_selection(title): st.session_state['movie'] = title

st.title("Movie Recommendation System")
st.markdown("Item-based filtering using user ratings from the MovieLens 32M dataset | Movie data from TMDB | Up to: 2023")

idx = int(df[df['title'] == st.session_state['movie']].index[0]) if st.session_state['movie'] in df['title'].values else None
selected = st.selectbox("Select a reference movie:", df['title'].values, index=idx, placeholder="Type a title (e.g. Inception)")
go_btn = st.button("Start", type="primary")

# 5. MOTEUR DE RECOMMANDATION
if selected and (go_btn or st.session_state['movie']):
    row = df[df['title'] == selected].iloc[0]
    info = get_details(row['tmdbId'])
    
    st.divider()
    
    # Hero Section
    c1, c2 = st.columns([1, 3])
    with c1: st.image(info['poster'], use_container_width=True)
    with c2:
        st.subheader(selected)
        st.caption(f"Year: {info['year']} | Runtime: {info['runtime']} | Genres: {info['genres']}")
        st.write(f"**TMDB Rating:** {info['rating']}/10")
        st.write(f"**Synopsis:** {info['overview']}")
        
        if info['streaming']:
            st.markdown("**Available on (FR):**")
            st.markdown("".join([f'<img src="{p["logo"]}" style="width:50px; margin-right:10px; border-radius:8px;" title="{p["name"]}">' for p in info['streaming']]), unsafe_allow_html=True)

    # Recommandations (KNN)
    st.subheader("Fans also liked:")
    distances, indices = model.kneighbors(matrix[row['matrice_id']], n_neighbors=6)
    cols = st.columns(5)
    
    for i, col in enumerate(cols):
        neighbor_id = indices.flatten()[i+1]
        neighbor_row = df[df['matrice_id'] == neighbor_id].iloc[0]
        neighbor_title = neighbor_row['title'] 
        n_info = get_details(neighbor_row['tmdbId'])
        
        with col:
            st.image(n_info['poster'], use_container_width=True)
            
            # Titre
            st.markdown(f"""
            <div style="height: 50px; display: flex; align-items: center; justify-content: center; text-align: center; font-weight: bold; font-size: 14px; margin-bottom: 5px; line-height: 1.2; overflow: hidden; text-overflow: ellipsis;">
                {neighbor_title}
            </div>
            """, unsafe_allow_html=True)
            
            # Barre de progression
            match = int((1 - distances.flatten()[i+1]) * 100)
            st.progress(match)
            
            # --- BLOC TEXTE AJUSTÉ ---
            # margin-top: 0px -> Colle le "Match" juste sous la barre
            # margin-top: 5px (sur la 2e div) -> Éloigne le Rating du Match
            st.markdown(f"""
            <div style="text-align: center; font-size: 14px; color: #555;">
                <div style="margin-top: 0px; margin-bottom: 0px;">Match: {match}%</div>
                <div style="margin-top: 6px; font-size: 13px; color: #777;">⭐ {n_info['rating']}</div>
            </div>
            """, unsafe_allow_html=True)

            # --- LOGOS STREAMING ---
            logos_html = ""
            if n_info and n_info['streaming']:
                logos_html = "".join([f'<img src="{p["logo"]}" style="width:35px; margin: 0 4px; border-radius:5px;" title="{p["name"]}">' for p in n_info['streaming']])
            
            # margin-top: 10px -> Éloigne les logos du Rating
            st.markdown(f"""
            <div style="height: 45px; display: flex; align-items: center; justify-content: center; margin-top: 10px; margin-bottom: 5px;">
                {logos_html}
            </div>
            """, unsafe_allow_html=True)

            # --- BOUTON CENTRÉ ---
            # use_container_width=True -> Force le bouton à prendre toute la largeur (donc centré)
            st.button("Search this movie", key=f"btn_{neighbor_id}", on_click=update_selection, args=(neighbor_title,), use_container_width=True)