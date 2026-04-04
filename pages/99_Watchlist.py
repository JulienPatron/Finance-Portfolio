import streamlit as st
import pandas as pd
import requests
import gspread
import datetime
import ast

st.set_page_config(page_title="Ma Watchlist", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    .movie-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
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
    .synopsis-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.85);
        color: white;
        opacity: 0;
        transition: opacity 0.3s ease;
        padding: 12px;
        font-size: 13px;
        line-height: 1.4em;
        overflow-y: auto;
        text-align: justify;
    }
    .poster-container:hover .synopsis-overlay {
        opacity: 1;
    }
    .synopsis-overlay::-webkit-scrollbar {
        width: 5px;
    }
    .synopsis-overlay::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 5px;
    }
    .movie-title { 
        font-weight: bold; 
        font-size: 16px; 
        line-height: 1.2em;
        height: 2.4em;
        margin-bottom: 5px; 
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .movie-meta { 
        font-size: 13px; 
        color: #555; 
        line-height: 1.4em;
        height: 2.8em;
        margin-bottom: 10px;
        overflow: hidden;
    }
    .streaming-container {
        height: 30px;
        margin-bottom: 5px;
        display: flex;
        flex-wrap: nowrap;
        overflow: hidden;
        align-items: center;
    }
    .provider-logo { 
        height: 30px; 
        width: 30px; 
        margin-right: 5px; 
        border-radius: 5px; 
        object-fit: cover;
        flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_google_sheet(worksheet_name="sheet1"):
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        sh = gc.open("Streamlit_Watchlist")
        return sh.sheet1 if worksheet_name == "sheet1" else sh.worksheet(worksheet_name)
    except Exception as e:
        st.error(f"Erreur de connexion Google Sheets : {e}")
        st.stop()

if st.session_state.get("admin_mode", False):
    worksheet_name = "sheet1"
elif st.session_state.get("user_mode") == "irene":
    worksheet_name = "irene"
else:
    st.error("Accès non autorisé.")
    st.stop()

sheet = get_google_sheet(worksheet_name)

TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
OMDB_API_KEY = st.secrets["OMDB_API_KEY"]

# Mise en cache du dictionnaire TMDB liant les URL des logos aux noms des plateformes
@st.cache_data(ttl=86400, show_spinner=False)
def get_providers_mapping():
    url = f"https://api.themoviedb.org/3/watch/providers/movie?api_key={TMDB_API_KEY}&language=fr-FR"
    data = requests.get(url).json()
    mapping = {}
    for p in data.get('results', []):
        full_url = f"https://image.tmdb.org/t/p/original{p['logo_path']}"
        mapping[full_url] = p['provider_name']
    return mapping

def search_movies(query):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}&language=fr-FR&page=1"
    response = requests.get(url).json()
    return response.get('results', [])[:10]

def get_movie_details(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=fr-FR&append_to_response=watch/providers"
    data = requests.get(url).json()
    
    providers_data = data.get('watch/providers', {}).get('results', {}).get('FR', {})
    flatrate = providers_data.get('flatrate', [])
    free = providers_data.get('free', [])
    plateformes = flatrate + free
    
    streaming_logos = [f"https://image.tmdb.org/t/p/original{p['logo_path']}" for p in plateformes]
    
    imdb_id = data.get('imdb_id')
    note_finale = "N/A"
    
    if imdb_id:
        omdb_url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
        try:
            omdb_data = requests.get(omdb_url).json()
            note_finale = omdb_data.get('imdbRating', "N/A")
        except:
            pass
            
    if note_finale == "N/A":
        raw_tmdb = data.get('vote_average', 0)
        note_finale = str(round(raw_tmdb, 1)) if raw_tmdb > 0 else "N/A"
    
    synopsis = data.get('overview', '')
    if not synopsis:
        synopsis = "Aucun synopsis disponible."
    
    return {
        "tmdb_id": tmdb_id,
        "titre": data.get('title'),
        "annee": data.get('release_date', '????')[:4],
        "duree": f"{data.get('runtime', 0)//60}h {data.get('runtime', 0)%60:02d}m",
        "genres": ", ".join([g['name'] for g in data.get('genres', [])][:3]),
        "note": note_finale,
        "poster_url": f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}" if data.get('poster_path') else "https://via.placeholder.com/300x450?text=No+Image",
        "streaming": str(streaming_logos),
        "date_ajout": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "synopsis": synopsis
    }

st.title("Ma Watchlist")

search_query = st.text_input("Titre du film :", placeholder="Rechercher un film...")

if search_query != st.session_state.get("last_query", ""):
    st.session_state["last_query"] = search_query
    st.session_state["hide_results"] = False

if search_query and not st.session_state.get("hide_results", False):
    results = search_movies(search_query)
    if results:
        st.markdown("**Résultats :**")
        existing_ids = {str(r.get('tmdb_id', '')) for r in sheet.get_all_records()}

        cols = st.columns(5)
        for i, movie in enumerate(results):
            with cols[i % 5]:
                titre = movie.get('title', 'Titre inconnu')
                annee = movie.get('release_date', '????')[:4]
                vo = movie.get('original_title', '')
                votes = movie.get('vote_count', 0)
                poster_path = movie.get('poster_path')
                poster_url = f"https://image.tmdb.org/t/p/w185{poster_path}" if poster_path else "https://via.placeholder.com/185x278?text=N/A"
                tmdb_id = movie['id']
                already_in = str(tmdb_id) in existing_ids

                st.markdown(f'<div style="position:relative;width:100%;padding-bottom:150%;border-radius:5px;overflow:hidden;margin-bottom:10px;"><img src="{poster_url}" style="position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;"></div>', unsafe_allow_html=True)
                vo_line = f'<div style="font-size:11px;color:#888;font-style:italic;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{vo}</div>' if vo and vo != titre else '<div style="font-size:11px;">&nbsp;</div>'
                st.markdown(f'''<div style="font-weight:bold;font-size:13px;line-height:1.3em;height:2.6em;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;">{titre} ({annee})</div>{vo_line}<div style="font-size:12px;color:#555;">{votes} avis</div>''', unsafe_allow_html=True)

                if already_in:
                    st.button("Déjà ajouté", key=f"add_{tmdb_id}", disabled=True, use_container_width=True)
                else:
                    if st.button("Ajouter", key=f"add_{tmdb_id}", use_container_width=True):
                        details = get_movie_details(tmdb_id)
                        row_to_insert = [
                            details["tmdb_id"], details["titre"], details["annee"],
                            details["duree"], details["genres"], details["note"],
                            details["poster_url"], details["streaming"], details["date_ajout"],
                            details["synopsis"]
                        ]
                        sheet.append_row(row_to_insert)
                        st.session_state["hide_results"] = True
                        st.rerun()

st.divider()

records = sheet.get_all_records()

if not records:
    st.info("La Watchlist est vide.")
else:
    df = pd.DataFrame(records)
    df['sheet_row'] = df.index + 2 

    # --- Section Tri et Filtres ---
    col1, col2, col3 = st.columns(3)

    with col1:
        sort_option = st.selectbox(
            "Trier par :",
            ["Date d'ajout (Plus récents d'abord)", "Note (Décroissant)", "Année de sortie (Récent d'abord)", "Ordre alphabétique"]
        )

    with col2:
        plateformes_cibles = ["Netflix", "Amazon Prime", "HBO / Max", "Canal+"]
        plateformes_filtre = st.multiselect("Filtrer par plateforme :", plateformes_cibles)

    with col3:
        all_genres = sorted(set(
            g.strip()
            for genres_str in df['genres'].dropna()
            for g in str(genres_str).split(',')
            if g.strip()
        ))
        genres_filtre = st.multiselect("Filtrer par genre :", all_genres)

    # 1. Application du Filtre de Plateforme
    if plateformes_filtre:
        providers_map = get_providers_mapping()
        
        def match_platform(row_logos_str):
            try:
                logos = ast.literal_eval(row_logos_str)
                for logo in logos:
                    provider_name = providers_map.get(logo, "")
                    for filtre in plateformes_filtre:
                        if filtre == "Netflix" and "Netflix" in provider_name:
                            return True
                        elif filtre == "Amazon Prime" and "Prime" in provider_name:
                            return True
                        elif filtre == "Canal+" and "Canal" in provider_name:
                            return True
                        elif filtre == "HBO / Max" and ("Max" in provider_name or "HBO" in provider_name or "Warner" in provider_name):
                            return True
                return False
            except:
                return False
                
        df = df[df['streaming'].apply(match_platform)]

    # 2. Filtre par genre
    if genres_filtre:
        def match_genre(genres_str):
            film_genres = [g.strip() for g in str(genres_str).split(',')]
            return any(g in film_genres for g in genres_filtre)
        df = df[df['genres'].apply(match_genre)]

    # Application du Tri
    if sort_option == "Date d'ajout (Plus récents d'abord)":
        df = df.sort_values(by='date_ajout', ascending=False)
    elif sort_option == "Note (Décroissant)":
        df['note_num'] = pd.to_numeric(df['note'], errors='coerce')
        df = df.sort_values(by='note_num', ascending=False)
    elif sort_option == "Année de sortie (Récent d'abord)":
        df = df.sort_values(by='annee', ascending=False)
    elif sort_option == "Ordre alphabétique":
        df = df.sort_values(by='titre', ascending=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if df.empty:
        st.info("Aucun film ne correspond à cette sélection.")
    else:
        cols = st.columns(4)
        
        for i, (_, row) in enumerate(df.iterrows()):
            col = cols[i % 4]
            
            with col:
                with st.container(border=True):
                    note_ui = row["note"]
                    
                    synopsis_ui = row.get("synopsis", "")
                    if pd.isna(synopsis_ui) or str(synopsis_ui).strip() == "":
                        synopsis_ui = "Synopsis non disponible pour ce film."

                    try:
                        logos = ast.literal_eval(row['streaming'])
                        if logos:
                            logos_html = "".join([f'<img src="{logo}" class="provider-logo">' for logo in logos])
                        else:
                            logos_html = '<div style="color:#aaa; font-size:12px;">Aucun stream gratuit</div>'
                    except:
                        logos_html = '<div style="color:#aaa; font-size:12px;">Aucun stream gratuit</div>'

                    card_html = f"""
                    <div class="poster-container">
                        <img src="{row['poster_url']}" class="movie-poster">
                        <div class="synopsis-overlay">{synopsis_ui}</div>
                    </div>
                    <div class="movie-title">{row['titre']} ({row['annee']})</div>
                    <div class="movie-meta">{note_ui}/10 | {row['duree']}<br>{row['genres']}</div>
                    <div class="streaming-container">{logos_html}</div>
                    """
                    
                    st.markdown(card_html, unsafe_allow_html=True)

                    if st.button("Marqué comme vu", key=f"del_{row['tmdb_id']}", use_container_width=True):
                        sheet.delete_rows(int(row['sheet_row']))
                        st.rerun()