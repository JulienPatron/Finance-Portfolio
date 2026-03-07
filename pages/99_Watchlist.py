import streamlit as st
import pandas as pd
import requests
import gspread
import datetime
import ast

st.set_page_config(page_title="My Watchlist", layout="wide")

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
    .provider-logo { width: 30px; margin: 0 4px; border-radius: 5px; }
    .movie-title { 
        font-weight: bold; 
        font-size: 16px; 
        margin-bottom: 5px; 
        min-height: 48px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .movie-meta { 
        font-size: 13px; 
        color: #555; 
        margin-bottom: 10px;
        min-height: 40px;
    }
    .streaming-container {
        min-height: 35px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_google_sheet():
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        sh = gc.open("Streamlit_Watchlist").sheet1
        return sh
    except Exception as e:
        st.error(f"Erreur de connexion Google Sheets : {e}")
        st.stop()

sheet = get_google_sheet()

TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

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
    
    return {
        "tmdb_id": tmdb_id,
        "titre": data.get('title'),
        "annee": data.get('release_date', '????')[:4],
        "duree": f"{data.get('runtime', 0)//60}h {data.get('runtime', 0)%60:02d}m",
        "genres": ", ".join([g['name'] for g in data.get('genres', [])][:3]),
        "note": int(data.get('vote_average', 0) * 10),
        "poster_url": f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}" if data.get('poster_path') else "https://via.placeholder.com/300x450?text=No+Image",
        "streaming": str(streaming_logos),
        "date_ajout": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

st.title("Ma Watchlist")

search_query = st.text_input("Titre du film :", placeholder="Rechercher un film...")

if search_query:
    results = search_movies(search_query)
    if results:
        options = {f"{m['title']} ({m.get('release_date', '????')[:4]})": m['id'] for m in results}
        options_list = ["-- Sélectionner un film --"] + list(options.keys())
        
        selected_movie = st.selectbox("Résultats :", options_list)
        
        if selected_movie != "-- Sélectionner un film --":
            tmdb_id = options[selected_movie]
            records = sheet.get_all_records()
            
            if any(str(r.get('tmdb_id', '')) == str(tmdb_id) for r in records):
                st.warning("Ce film est déjà dans la Watchlist.")
            else:
                details = get_movie_details(tmdb_id)
                row_to_insert = [
                    details["tmdb_id"], details["titre"], details["annee"], 
                    details["duree"], details["genres"], details["note"], 
                    details["poster_url"], details["streaming"], details["date_ajout"]
                ]
                sheet.append_row(row_to_insert)
                st.success(f"{details['titre']} ajouté avec succès.")
                st.rerun()

st.divider()

records = sheet.get_all_records()

if not records:
    st.info("La Watchlist est vide.")
else:
    df = pd.DataFrame(records)
    df['sheet_row'] = df.index + 2 

    sort_option = st.selectbox(
        "Trier par :", 
        ["Date d'ajout (Plus récents d'abord)", "Note TMDB (Décroissant)", "Année de sortie (Récent d'abord)", "Ordre alphabétique"]
    )
    
    if sort_option == "Date d'ajout (Plus récents d'abord)":
        df = df.sort_values(by='date_ajout', ascending=False)
    elif sort_option == "Note TMDB (Décroissant)":
        df['note'] = pd.to_numeric(df['note'], errors='coerce')
        df = df.sort_values(by='note', ascending=False)
    elif sort_option == "Année de sortie (Récent d'abord)":
        df = df.sort_values(by='annee', ascending=False)
    elif sort_option == "Ordre alphabétique":
        df = df.sort_values(by='titre', ascending=True)

    cols = st.columns(4)
    
    for i, (_, row) in enumerate(df.iterrows()):
        col = cols[i % 4]
        
        with col:
            try:
                raw_note = float(row["note"])
                note_ui = round(raw_note / 10, 1) if raw_note > 10 else round(raw_note, 1)
            except:
                note_ui = row["note"]

            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            st.image(row['poster_url'], use_container_width=True)
            st.markdown(f'<div class="movie-title">{row["titre"]} ({row["annee"]})</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="movie-meta">{note_ui}/10 | {row["duree"]}<br>{row["genres"]}</div>', unsafe_allow_html=True)
            
            try:
                logos = ast.literal_eval(row['streaming'])
                if logos:
                    logos_html = "".join([f'<img src="{logo}" class="provider-logo">' for logo in logos])
                    st.markdown(f'<div class="streaming-container">{logos_html}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="streaming-container" style="color:#aaa; font-size:12px;">Aucun stream gratuit</div>', unsafe_allow_html=True)
            except:
                st.markdown('<div class="streaming-container" style="color:#aaa; font-size:12px;">Aucun stream gratuit</div>', unsafe_allow_html=True)

            if st.button("Marqué comme vu", key=f"del_{row['tmdb_id']}", use_container_width=True):
                sheet.delete_rows(int(row['sheet_row']))
                st.rerun()
                
            st.markdown('</div>', unsafe_allow_html=True)

import time

st.divider()

if st.button("Importer les 107 films d'un coup"):
    films_a_ajouter = [
        "12 Years a Slave", "120 battements par minute", "1917", "2001 : l'odyssée de l'espace",
        "Aftersun", "Akira", "Anatomie d'une chute", "Anora", "Apocalypse Now", "Argo",
        "Arnaque américaine", "Au revoir là-haut", "Aviator", "Bac Nord", "Bagdad Café",
        "Barry Lyndon", "Before Sunrise", "Boîte noire", "CODA", "District 9", "Dunkerque",
        "Délire Express", "Démineurs", "El Camino : Un film Breaking Bad", "Enemy",
        "Enron: The Smartest Guys in the Room", "Fargo", "First Man - le premier homme sur la Lune",
        "Get Out", "Gone Girl", "Hamnet", "Heat", "Il faut sauver le soldat Ryan", "Incendies",
        "L'Armée des ombres", "L'Innocence", "L'Étrange Histoire de Benjamin Button", "La Chasse",
        "La Cité de la Peur : une comédie familiale", "La Folle Histoire de l'espace", "La Haine",
        "La Ligne verte", "La Liste de Schindler", "La Tête haute", "La Zone d'intérêt",
        "Le Cercle des poètes disparus", "Le Fabuleux destin d'Amélie Poulain", "Le Garçon au pyjama rayé",
        "Le Pianiste", "Le Talentueux Mr Ripley", "Le garçon qui dompta le vent", "Le pont des espions",
        "Le secret de Brokeback Mountain", "Les Enfants du temps", "Les Fils de l'homme", "Limitless",
        "Lion", "Lost in Translation", "Mademoiselle", "Marty Supreme", "McFarland, USA", "Memento",
        "Memories of murder", "Midsommar", "Moonlight", "Mystic River", "Mémoires de nos pères",
        "Night Call", "No Country for Old Men", "Old Boy", "Onoda, 10 000 nuits dans la jungle",
        "Paprika", "Past Lives - Nos vies d'avant", "Portrait de la jeune fille en feu", "Primer",
        "Prisoners", "Pusher", "Requiem pour un massacre", "Schumacher", "Senna", "Shutter Island",
        "Spies of Terror", "Split", "Taxi Driver", "The Apprentice", "The Artist", "The Brutalist",
        "The Constant Gardener", "The Father", "The Gentlemen", "The Irishman", "The Nice Guys",
        "The Outsider", "The Power of the Dog", "The Revenant", "The Spectacular Now", "There Will Be Blood",
        "Thunderbolts*", "Top secret !", "Un homme d'exception", "Un parfait inconnu",
        "Une bataille après l'autre", "Vice-Versa", "Voyage au bout de l'enfer", "Warrior",
        "When Life Gives You Tangerines", "Wicked", "Yi Yi", "À vif !"
    ]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    records = sheet.get_all_records()
    existing_ids = [str(r.get('tmdb_id', '')) for r in records]
    lignes_a_inserer = []
    
    for i, titre in enumerate(films_a_ajouter):
        status_text.text(f"Traitement de : {titre} ({i+1}/{len(films_a_ajouter)})")
        
        results = search_movies(titre)
        if results:
            tmdb_id = results[0]['id']
            
            if str(tmdb_id) not in existing_ids:
                details = get_movie_details(tmdb_id)
                lignes_a_inserer.append([
                    details["tmdb_id"], details["titre"], details["annee"], 
                    details["duree"], details["genres"], details["note"], 
                    details["poster_url"], details["streaming"], details["date_ajout"]
                ])
                existing_ids.append(str(tmdb_id))
        
        progress_bar.progress((i + 1) / len(films_a_ajouter))
        time.sleep(0.25)
        
    if lignes_a_inserer:
        status_text.text("Envoi vers Google Sheets...")
        sheet.append_rows(lignes_a_inserer)
        st.success("Importation terminée.")
        time.sleep(2)
        st.rerun()
    else:
        st.info("Tous les films sont déjà dans la liste.")