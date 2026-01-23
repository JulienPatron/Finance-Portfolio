import streamlit as st
import pandas as pd
import pickle
import requests
import os
import gzip

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Ciné Match - AI Engine",
    layout="wide"
)

# --- CSS PERSONNALISÉ (Style Pro) ---
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
# 2. CHARGEMENT DES DONNÉES & API
# ==============================================================================

# Récupération sécurisée de la clé API
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except FileNotFoundError:
    st.error("⚠️ Erreur : Fichier .streamlit/secrets.toml introuvable.")
    st.stop()

@st.cache_resource
def load_engine():
    """Charge le modèle et la matrice depuis les fichiers compressés (.gz)."""
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
        st.error(f"Erreur de chargement : {e}")
        return None, None, None

def fetch_movie_details(tmdb_id):
    """
    Récupère les détails complets (Affiche, Synopsis, Date, Note) via TMDB.
    """
    if pd.isna(tmdb_id):
        return None
    
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}&language=fr-FR"
    try:
        response = requests.get(url, timeout=1.5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# Chargement du moteur
model, matrix, df_movies = load_engine()

if model is None:
    st.stop()

# ==============================================================================
# 3. GESTION DU "RABBIT HOLE" (Session State)
# ==============================================================================

# Si on n'a pas encore de film sélectionné en mémoire, on initialise
if 'selected_movie_name' not in st.session_state:
    st.session_state['selected_movie_name'] = None

# Fonction pour mettre à jour la sélection quand on clique sur "Explorer"
def set_movie(movie_title):
    st.session_state['selected_movie_name'] = movie_title

# ==============================================================================
# 4. INTERFACE PRINCIPALE
# ==============================================================================

st.title("Movie Recommendation System")
st.markdown("From MovieLens 32M Dataset")

# --- BARRE DE RECHERCHE INTELLIGENTE ---
# On cherche l'index du film stocké en session (s'il existe) pour pré-remplir la box
index_to_select = None
if st.session_state['selected_movie_name'] in df_movies['title'].values:
    index_to_select = int(df_movies[df_movies['title'] == st.session_state['selected_movie_name']].index[0])

selected_movie = st.selectbox(
    "🔍 Recherchez un film de référence :",
    df_movies['title'].values,
    index=index_to_select,
    placeholder="Tapez un titre (ex: Inception)...",
)

# Bouton d'action (ou auto-run si un film est sélectionné via le Rabbit Hole)
start_analysis = st.button("Lancer l'analyse", type="primary")

# ==============================================================================
# 5. MOTEUR DE RECOMMANDATION & AFFICHAGE
# ==============================================================================

if selected_movie and (start_analysis or st.session_state['selected_movie_name']):
    
    # 1. Récupération des infos du film sélectionné (HERO SECTION)
    idx = df_movies[df_movies['title'] == selected_movie].index[0]
    tmdb_id_source = df_movies.iloc[idx]['tmdbId']
    matrice_id = df_movies.iloc[idx]['matrice_id']
    
    source_details = fetch_movie_details(tmdb_id_source)
    
    st.divider()
    
    # --- HERO SECTION (Le film choisi) ---
    col_hero_img, col_hero_txt = st.columns([1, 3])
    
    with col_hero_img:
        if source_details and source_details.get('poster_path'):
            st.image(f"https://image.tmdb.org/t/p/w500{source_details['poster_path']}", use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
            
    with col_hero_txt:
        st.subheader(f"{selected_movie}")
        if source_details:
            date_sortie = source_details.get('release_date', 'Inconnue')[:4]
            note = round(source_details.get('vote_average', 0), 1)
            overview = source_details.get('overview', 'Pas de résumé disponible.')
            
            st.caption(f"Année : {date_sortie} | Note TMDB : {note}/10")
            st.write(f"**Synopsis :** {overview}")
        
        st.markdown("### Films recommandés :")

    # 2. CALCUL KNN
    distances, indices = model.kneighbors(matrix[matrice_id], n_neighbors=6)
    
    # 3. AFFICHAGE DES RÉSULTATS (GRID)
    st.write("") # Petit espace
    cols = st.columns(5)
    
    # On boucle sur les 5 voisins (on ignore le premier qui est le film lui-même)
    for i, col in enumerate(cols):
        neighbor_idx = indices.flatten()[i+1] # i+1 pour sauter le film source
        distance = distances.flatten()[i+1]
        similarity = 1 - distance # Conversion Distance -> Similarité
        
        # Récupération des data du voisin
        match = df_movies[df_movies['matrice_id'] == neighbor_idx]
        
        if not match.empty:
            neighbor_data = match.iloc[0]
            neighbor_title = neighbor_data['title']
            neighbor_details = fetch_movie_details(neighbor_data['tmdbId'])
            
            with col:
                # Affiche Image
                if neighbor_details and neighbor_details.get('poster_path'):
                    st.image(f"https://image.tmdb.org/t/p/w500{neighbor_details['poster_path']}", use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
                
                # Titre & Année
                year = "????"
                if neighbor_details and neighbor_details.get('release_date'):
                    year = neighbor_details.get('release_date')[:4]
                
                st.markdown(f"**{neighbor_title}** ({year})")
                
                # Jauge de similarité
                st.progress(int(similarity * 100))
                st.caption(f"Match : {int(similarity * 100)}%")
                
                # BOUTON RABBIT HOLE 🐰
                # Si on clique, on met à jour le session_state et on recharge
                if st.button("Search this movie", key=f"btn_{neighbor_idx}"):
                    set_movie(neighbor_title)
                    st.rerun()

elif not selected_movie:
    st.info("👈 Sélectionnez un film dans le menu ou tapez un titre pour commencer l'exploration.")