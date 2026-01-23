import streamlit as st
import pandas as pd
import pickle
import requests
import os
import gzip  # <--- Indispensable pour lire vos fichiers compressés

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Movie Recommendation Engine - Julien Patron",
    layout="wide"
)

# --- CSS INSTITUTIONNEL (Cohérence avec Portfolio Optimizer) ---
st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CONFIGURATION & CHARGEMENT
# ==========================================

# Récupération sécurisée de la clé API via st.secrets
# (Fonctionne en local grâce au fichier .streamlit/secrets.toml)
# (Fonctionnera en ligne grâce à la config du Cloud)
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except FileNotFoundError:
    st.error("⚠️ Erreur : La clé API n'a pas été trouvée dans les secrets.")
    st.stop()

@st.cache_resource
def load_engine():
    """
    Loads the KNN model and sparse matrix from compressed disk files (.gz).
    Cached to prevent reloading on every interaction.
    """
    # Noms des fichiers (tels qu'ils sont à la racine maintenant)
    paths = {
        "model": "modele_knn.pkl.gz",     
        "matrix": "matrice_sparse.pkl.gz", 
        "data": "liste_films_final.csv"
    }
    
    # Gestion des chemins : Si le script ne trouve pas les fichiers,
    # c'est qu'il cherche dans 'pages/', donc on remonte d'un cran.
    if not os.path.exists(paths["data"]):
        paths = {k: f"../{v}" for k, v in paths.items()}

    try:
        # 1. Lecture du Modèle (Compressé)
        with gzip.open(paths["model"], 'rb') as f:
            model = pickle.load(f)
            
        # 2. Lecture de la Matrice (Compressée)
        with gzip.open(paths["matrix"], 'rb') as f:
            matrix = pickle.load(f)
            
        # 3. Lecture des Données (CSV standard non compressé)
        df = pd.read_csv(paths["data"])
        
        return model, matrix, df
        
    except FileNotFoundError as e:
        st.error(f"System Error: Files not found. Ensure .gz and .csv files are at the project root. Debug: {e}")
        return None, None, None

def fetch_poster(tmdb_id):
    """Fetches movie poster URL via TMDB API."""
    if pd.isna(tmdb_id):
        return "https://via.placeholder.com/400x600?text=No+Image"
    
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        response = requests.get(url, timeout=1.5)
        if response.status_code == 200:
            data = response.json()
            path = data.get('poster_path')
            if path:
                return f"https://image.tmdb.org/t/p/w500{path}"
    except:
        pass
    return "https://via.placeholder.com/400x600?text=Unavailable"

# ==============================================================================
# 2. INTERFACE UTILISATEUR
# ==============================================================================

st.title("Algorithmic Recommendation Engine")
st.markdown("Item-Based Collaborative Filtering (KNN) on MovieLens 32M Dataset.")
st.markdown("---")

# Chargement du moteur IA
with st.spinner('Initializing AI Engine (Decompressing models)...'):
    model, matrix, df_movies = load_engine()

if model is None:
    st.stop()

# Layout: 2 colonnes (Recherche à gauche, Espace à droite)
col_search, col_info = st.columns([1, 2])

with col_search:
    st.subheader("Input Parameters")
    selected_movie = st.selectbox(
        "Select a Reference Movie:",
        df_movies['title'].values,
        index=None,
        placeholder="Type to search..."
    )
    
    run_btn = st.button("Generate Recommendations", type="primary")

# ==============================================================================
# 3. MOTEUR DE RECOMMANDATION
# ==============================================================================

if run_btn and selected_movie:
    st.divider()
    st.subheader(f"Analysis: Users who liked '{selected_movie}' also liked:")
    
    # 1. Récupération de l'index interne
    idx = df_movies[df_movies['title'] == selected_movie].index[0]
    matrice_id = df_movies.iloc[idx]['matrice_id']
    
    # 2. Inférence (Recherche des voisins)
    # n_neighbors=6 car le premier résultat est toujours le film lui-même
    distances, indices = model.kneighbors(matrix[matrice_id], n_neighbors=6)
    
    # 3. Affichage en Grille
    cols = st.columns(5)
    
    for i, col in enumerate(cols):
        # On commence à i+1 pour ignorer le film d'origine
        neighbor_idx = indices.flatten()[i+1]
        
        # Récupération des métadonnées du voisin
        match = df_movies[df_movies['matrice_id'] == neighbor_idx]
        
        if not match.empty:
            film_data = match.iloc[0]
            with col:
                # Affichage propre type "Carte"
                poster_url = fetch_poster(film_data['tmdbId'])
                st.image(poster_url, use_container_width=True)
                st.markdown(f"**{film_data['title']}**")
                
                # Optionnel : Afficher la distance (similitude inverse)
                similarity = 1 - distances.flatten()[i+1]
                st.caption(f"Similarity Score: {similarity:.2f}")

elif run_btn and not selected_movie:
    st.warning("⚠️ Please select a movie to start the analysis.")