import streamlit as st
import pandas as pd
import pickle, requests, os, gzip
import gc

# 1. CONFIGURATION
st.set_page_config(page_title="Project Portfolio", layout="wide")
st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# 2. CHARGEMENT OPTIMISÉ (CACHE PERSISTANT)
@st.cache_resource(show_spinner=False)
def load_data():
    try:
        # Chemins relatifs adaptés à la structure multipage standard
        # On suppose que les fichiers sont dans le même dossier ou un dossier data
        # Ajustez les chemins si besoin selon votre structure exacte
        files = {"model": "modele_knn.pkl.gz", "matrix": "matrice_sparse.pkl.gz", "db": "liste_films_final.csv"}
        
        # Petit hack pour gérer les chemins si on lance depuis la racine ou le dossier pages
        if not os.path.exists(files["db"]): 
            # Tentative de remonter d'un cran si on est dans pages/
            files = {k: f"../{v}" if not os.path.exists(v) else v for k, v in files.items()}
            
            # Si toujours pas trouvé, chercher dans un dossier data/ (convention courante)
            if not os.path.exists(files["db"]):
                 files = {k: f"data/{v}" for k, v in files.items()}

        with gzip.open(files["model"], 'rb') as f: 
            model = pickle.load(f)
        
        with gzip.open(files["matrix"], 'rb') as f: 
            matrix = pickle.load(f)

        df = pd.read_csv(files["db"])
        
        return model, matrix, df
    except Exception as e:
        return None, None, None

with st.spinner("Loading Movie Database..."):
    model, matrix, df = load_data()

if model is None:
    st.error("Error loading data files. Please check paths.")
    st.stop()

# 3. FONCTION API TMDB (AVEC CACHE)
# Mise en cache des appels API pour accélérer la navigation si on revient sur un film
@st.cache_data(ttl=3600, show_spinner=False) 
def get_details(tmdb_id):
    if pd.isna(tmdb_id): return None
    # Utilisation de st.secrets pour la clé API, avec fallback safe
    api_key = st.secrets.get("TMDB_API_KEY", "")
    if not api_key: return None
    
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}&language=en-US&append_to_response=watch/providers"
    
    try:
        data = requests.get(url, timeout=1.5).json() # Timeout légèrement augmenté
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

# 4. INTERFACE
if 'movie' not in st.session_state: st.session_state['movie'] = None

def update_selection(title): st.session_state['movie'] = title

st.title("Movie Recommendation System")
st.markdown("Item-based filtering using user ratings from the MovieLens 32M dataset.")

# Selectbox optimisée
idx = int(df[df['title'] == st.session_state['movie']].index[0]) if st.session_state['movie'] in df['title'].values else None
selected = st.selectbox("Select a reference movie:", df['title'].values, index=idx, placeholder="Type a title...", key="main_select")

# Bouton moins intrusif
if st.button("Get Recommendations", type="primary", use_container_width=True) or st.session_state['movie']:
    if selected:
        row = df[df['title'] == selected].iloc[0]
        info = get_details(row['tmdbId'])
        
        if info is None:
            info = {"poster": "https://via.placeholder.com/300x450?text=Error", "year": "????", "runtime": "N/A", "genres": "N/A", "rating": "N/A", "overview": "Unavailable", "streaming": []}

        st.divider()
        
        # Hero Section
        c1, c2 = st.columns([1, 3])
        with c1: st.image(info['poster'], use_container_width=True)
        with c2:
            st.subheader(f"{selected} ({info['year']})")
            st.caption(f"{info['genres']} • {info['runtime']}")
            st.write(f"**Rating:** {info['rating']}/10")
            st.write(info['overview'])
            
            if info['streaming']:
                st.write("**Available on:**")
                st.markdown("".join([f'<img src="{p["logo"]}" style="width:40px; margin-right:8px; border-radius:6px;" title="{p["name"]}">' for p in info['streaming']]), unsafe_allow_html=True)

        # Recommandations
        st.subheader("You might also like:")
        
        # KNN Query
        # Pas besoin de réimporter scikit-learn ici, l'objet 'model' a déjà les méthodes
        distances, indices = model.kneighbors(matrix[row['matrice_id']], n_neighbors=6)
        
        cols = st.columns(5)
        for i, col in enumerate(cols):
            # On saute le premier (0) car c'est le film lui-même
            neighbor_idx = i + 1
            if neighbor_idx >= len(indices.flatten()): break
            
            neighbor_id = indices.flatten()[neighbor_idx]
            # Vérification de l'index pour éviter crash
            matches = df[df['matrice_id'] == neighbor_id]
            if matches.empty: continue
                
            neighbor_row = matches.iloc[0]
            neighbor_title = neighbor_row['title']
            
            with col:
                n_info = get_details(neighbor_row['tmdbId'])
                if n_info is None: n_info = {"poster": "https://via.placeholder.com/300x450?text=Unavailable", "rating": "N/A"}
                
                st.image(n_info['poster'], use_container_width=True)
                st.markdown(f"**{neighbor_title}**", help=f"Match: {int((1-distances.flatten()[neighbor_idx])*100)}%")
                st.button("See details", key=f"btn_{neighbor_id}", on_click=update_selection, args=(neighbor_title,), use_container_width=True)