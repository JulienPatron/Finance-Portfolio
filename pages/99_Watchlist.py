import streamlit as st
import pandas as pd
import requests
import gspread
import datetime
import ast
import re
import json

st.set_page_config(page_title="Ma Watchlist", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
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
        top: 0; left: 0;
        width: 100%; height: 100%;
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
    .poster-container:hover .synopsis-overlay { opacity: 1; }
    .synopsis-overlay::-webkit-scrollbar { width: 5px; }
    .synopsis-overlay::-webkit-scrollbar-thumb { background: #888; border-radius: 5px; }
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
        height: 30px; width: 30px;
        margin-right: 5px;
        border-radius: 5px;
        object-fit: cover;
        flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Connexion Google Sheets ---

@st.cache_resource(show_spinner=False)
def get_google_sheet(worksheet_name):
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

# --- Fonctions API ---

@st.cache_data(ttl=86400, show_spinner=False)
def get_providers_mapping():
    url = f"https://api.themoviedb.org/3/watch/providers/movie?api_key={TMDB_API_KEY}&language=fr-FR"
    data = requests.get(url).json()
    return {
        f"https://image.tmdb.org/t/p/original{p['logo_path']}": p['provider_name']
        for p in data.get('results', [])
    }

def get_letterboxd_rating(imdb_id):
    if not imdb_id:
        return "N/A"
    try:
        url = f"https://letterboxd.com/imdb/{imdb_id}/"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        match = re.search(r'<script type="application/ld\+json">(.*?)</script>', r.text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            rating = data.get('aggregateRating', {}).get('ratingValue')
            if rating:
                return str(round(float(rating), 2))
    except:
        pass
    return "N/A"

def search_movies(query):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}&language=fr-FR&page=1"
    return requests.get(url).json().get('results', [])[:10]

def get_movie_details(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=fr-FR&append_to_response=watch/providers"
    data = requests.get(url).json()

    providers_data = data.get('watch/providers', {}).get('results', {}).get('FR', {})
    streaming_logos = [
        f"https://image.tmdb.org/t/p/original{p['logo_path']}"
        for p in providers_data.get('flatrate', []) + providers_data.get('free', [])
    ]

    imdb_id = data.get('imdb_id')
    note = "N/A"
    if imdb_id:
        try:
            omdb = requests.get(f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}").json()
            note = omdb.get('imdbRating', "N/A")
        except:
            pass
    if note == "N/A":
        avg = data.get('vote_average', 0)
        note = str(round(avg, 1)) if avg > 0 else "N/A"

    synopsis = data.get('overview', '') or "Aucun synopsis disponible."

    return {
        "tmdb_id": tmdb_id,
        "titre": data.get('title'),
        "annee": data.get('release_date', '????')[:4],
        "duree": f"{data.get('runtime', 0)//60}h {data.get('runtime', 0)%60:02d}m",
        "genres": ", ".join([g['name'] for g in data.get('genres', [])][:3]),
        "note": note,
        "poster_url": f"https://image.tmdb.org/t/p/w500{data['poster_path']}" if data.get('poster_path') else "https://via.placeholder.com/300x450?text=No+Image",
        "streaming": str(streaming_logos),
        "date_ajout": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "synopsis": synopsis,
        "note_letterboxd": get_letterboxd_rating(imdb_id)
    }

# --- UI ---

st.title("Ma Watchlist")

if "search_key" not in st.session_state:
    st.session_state["search_key"] = 0

search_query = st.text_input(
    "Titre du film :",
    placeholder="Rechercher un film...",
    key=f"search_{st.session_state['search_key']}"
)

TMDB_IDS_TO_RELOAD = [
    76203, 451945, 530915, 62, 965150, 149, 915935, 1064213, 28, 68734,
    168672, 430424, 637534, 3543, 3175, 76, 663260, 776503, 17654, 374720,
    10189, 12162, 559969, 181886, 13020, 275, 369972, 419430, 210577, 858024,
    949, 857, 46738, 15383, 1050035, 4922, 15097, 957, 406, 497, 424, 329819,
    467244, 207, 194, 14574, 1213, 491480, 296098, 142, 568160, 9693, 51876,
    334543, 153, 1317288, 228203, 77, 11423, 530385, 376867, 322, 3683,
    242582, 6977, 670, 695089, 666277, 531428, 14337, 146233, 855823, 58496,
    11324, 381288, 103, 1182047, 74643, 549509, 1985, 600354, 522627, 398978,
    290250, 199928, 600583, 157386, 7345, 986056, 8764, 453, 661539, 1054867,
    1022789, 11778, 59440, 402431, 25538, 295964, 2567, 290098, 103663, 82693,
    359940, 286217, 823452, 687163, 1327819
]

if st.session_state.get("admin_mode", False):
    if st.button("Reuploader tous les films", type="primary"):
        progress = st.progress(0, text="Récupération des données...")
        total = len(TMDB_IDS_TO_RELOAD)
        all_rows = []
        for i, tmdb_id in enumerate(TMDB_IDS_TO_RELOAD):
            details = get_movie_details(tmdb_id)
            all_rows.append([
                details["tmdb_id"], details["titre"], details["annee"],
                details["duree"], details["genres"], details["note"],
                details["poster_url"], details["streaming"], details["date_ajout"],
                details["synopsis"], details["note_letterboxd"]
            ])
            progress.progress((i + 1) / total, text=f"{details['titre']} ({i+1}/{total})")
        progress.progress(1.0, text="Insertion dans Google Sheets...")
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        sh = gc.open("Streamlit_Watchlist")
        ws = sh.sheet1 if worksheet_name == "sheet1" else sh.worksheet(worksheet_name)
        ws.append_rows(all_rows, value_input_option="USER_ENTERED")
        st.success("Tous les films ont été rechargés !")
        st.rerun()

if search_query:
    results = search_movies(search_query)
    if results:
        col_titre, col_fermer = st.columns([6, 1])
        with col_titre:
            st.markdown("**Résultats :**")
        with col_fermer:
            if st.button("Fermer ✕", use_container_width=True):
                st.session_state["search_key"] += 1
                st.rerun()

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

                st.markdown(
                    f'<div style="position:relative;width:100%;padding-bottom:150%;border-radius:5px;overflow:hidden;margin-bottom:10px;">'
                    f'<img src="{poster_url}" style="position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;"></div>',
                    unsafe_allow_html=True
                )
                vo_line = (
                    f'<div style="font-size:11px;color:#888;font-style:italic;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{vo}</div>'
                    if vo and vo != titre else
                    '<div style="font-size:11px;">&nbsp;</div>'
                )
                st.markdown(
                    f'<div style="font-weight:bold;font-size:13px;line-height:1.3em;height:2.6em;overflow:hidden;'
                    f'display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;">{titre} ({annee})</div>'
                    f'{vo_line}'
                    f'<div style="font-size:12px;color:#555;">{votes} avis</div>',
                    unsafe_allow_html=True
                )

                if str(tmdb_id) in existing_ids:
                    st.button("Déjà ajouté", key=f"add_{tmdb_id}", disabled=True, use_container_width=True)
                else:
                    if st.button("Ajouter", key=f"add_{tmdb_id}", use_container_width=True):
                        details = get_movie_details(tmdb_id)
                        sheet.append_row([
                            details["tmdb_id"], details["titre"], details["annee"],
                            details["duree"], details["genres"], details["note"],
                            details["poster_url"], details["streaming"], details["date_ajout"],
                            details["synopsis"], details["note_letterboxd"]
                        ])
                        st.rerun()

st.divider()

# --- Watchlist ---

records = sheet.get_all_records()

if not records:
    st.info("La Watchlist est vide.")
else:
    df = pd.DataFrame(records)
    df['sheet_row'] = df.index + 2

    col1, col2, col3 = st.columns(3)

    with col1:
        sort_option = st.selectbox("Trier par :", [
            "Date d'ajout (Plus récents d'abord)",
            "Note IMDB (Décroissant)",
            "Note Letterboxd (Décroissant)",
            "Année de sortie (Récent d'abord)",
            "Ordre alphabétique"
        ])

    with col2:
        plateformes_filtre = st.multiselect("Filtrer par plateforme :", ["Netflix", "Amazon Prime", "HBO / Max", "Canal+"])

    with col3:
        all_genres = sorted(set(
            g.strip()
            for genres_str in df['genres'].dropna()
            for g in str(genres_str).split(',')
            if g.strip()
        ))
        genres_filtre = st.multiselect("Filtrer par genre :", all_genres)

    if plateformes_filtre:
        providers_map = get_providers_mapping()
        def match_platform(row_logos_str):
            try:
                logos = ast.literal_eval(row_logos_str)
                for logo in logos:
                    name = providers_map.get(logo, "")
                    for f in plateformes_filtre:
                        if f == "Netflix" and "Netflix" in name: return True
                        if f == "Amazon Prime" and "Prime" in name: return True
                        if f == "Canal+" and "Canal" in name: return True
                        if f == "HBO / Max" and any(x in name for x in ["Max", "HBO", "Warner"]): return True
            except:
                pass
            return False
        df = df[df['streaming'].apply(match_platform)]

    if genres_filtre:
        df = df[df['genres'].apply(
            lambda s: any(g in [x.strip() for x in str(s).split(',')] for g in genres_filtre)
        )]

    if sort_option == "Date d'ajout (Plus récents d'abord)":
        df = df.sort_values('date_ajout', ascending=False)
    elif sort_option == "Note IMDB (Décroissant)":
        df['note_num'] = pd.to_numeric(df['note'], errors='coerce')
        df = df.sort_values('note_num', ascending=False)
    elif sort_option == "Note Letterboxd (Décroissant)":
        df['note_lb_num'] = pd.to_numeric(df.get('note_letterboxd', pd.Series(dtype=str)), errors='coerce')
        df = df.sort_values('note_lb_num', ascending=False)
    elif sort_option == "Année de sortie (Récent d'abord)":
        df = df.sort_values('annee', ascending=False)
    elif sort_option == "Ordre alphabétique":
        df = df.sort_values('titre', ascending=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if df.empty:
        st.info("Aucun film ne correspond à cette sélection.")
    else:
        cols = st.columns(4)
        for i, (_, row) in enumerate(df.iterrows()):
            with cols[i % 4]:
                with st.container(border=True):
                    synopsis_ui = str(row.get("synopsis", "")) if not pd.isna(row.get("synopsis", "")) else ""
                    synopsis_ui = synopsis_ui.strip() or "Synopsis non disponible pour ce film."

                    try:
                        logos = ast.literal_eval(row['streaming'])
                        logos_html = "".join([f'<img src="{l}" class="provider-logo">' for l in logos]) if logos else '<div style="color:#aaa;font-size:12px;">Aucun stream gratuit</div>'
                    except:
                        logos_html = '<div style="color:#aaa;font-size:12px;">Aucun stream gratuit</div>'

                    st.markdown(f"""
                    <div class="poster-container">
                        <img src="{row['poster_url']}" class="movie-poster">
                        <div class="synopsis-overlay">{synopsis_ui}</div>
                    </div>
                    <div class="movie-title">{row['titre']} ({row['annee']})</div>
                    <div class="movie-meta">IMDB {row['note']}/10 | LB {row.get('note_letterboxd', 'N/A')}/5 | {row['duree']}<br>{row['genres']}</div>
                    <div class="streaming-container">{logos_html}</div>
                    """, unsafe_allow_html=True)

                    if st.button("Marqué comme vu", key=f"del_{row['tmdb_id']}", use_container_width=True):
                        sheet.delete_rows(int(row['sheet_row']))
                        st.rerun()
