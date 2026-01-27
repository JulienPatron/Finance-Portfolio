import streamlit as st
import pandas as pd

# 1. CHARGEMENT OPTIMISÉ (Via CSV Kaggle)
@st.cache_data
def load_kaggle_data():
    """
    Charge et fusionne les CSV Kaggle pour créer le Master Dataset.
    """
    try:
        # On suppose que vous avez mis les csv dans un dossier 'data'
        races = pd.read_csv("data/races.csv")
        results = pd.read_csv("data/results.csv")
        drivers = pd.read_csv("data/drivers.csv")
        constructors = pd.read_csv("data/constructors.csv")
        
        # --- ÉTAPE 1 : FUSION (MERGE) ---
        # On colle les infos de la course (Année, Date) aux résultats
        df = pd.merge(results, races[['raceId', 'year', 'round', 'name', 'date']], on='raceId', how='left')
        
        # On colle les noms des pilotes
        df = pd.merge(df, drivers[['driverId', 'code', 'surname']], on='driverId', how='left')
        
        # On colle les noms des écuries
        df = pd.merge(df, constructors[['constructorId', 'name']], on='constructorId', how='left')
        
        # --- ÉTAPE 2 : NETTOYAGE ---
        # Renommer les colonnes pour que ce soit clair
        df = df.rename(columns={
            'year': 'Year',
            'round': 'Round',
            'name_x': 'GP_Name',      # Le nom du GP
            'name_y': 'Team',         # Le nom de l'équipe
            'code': 'Driver',         # VER, HAM...
            'surname': 'Surname',
            'positionOrder': 'Position' # PositionOrder est plus fiable que positionText
        })
        
        # On trie chronologiquement (CRUCIAL pour l'Elo)
        df = df.sort_values(by=['Year', 'Round', 'Position'])
        
        # Sélection des colonnes utiles
        final_df = df[['Year', 'Round', 'Date', 'GP_Name', 'Driver', 'Surname', 'Team', 'Position']]
        
        return final_df
        
    except FileNotFoundError:
        st.error("⚠️ Fichiers CSV introuvables. Placez races.csv, results.csv, drivers.csv et constructors.csv dans le dossier 'data'.")
        return None

# --- INTERFACE ---
st.title("🏎️ F1 Elo : Ingestion Kaggle")

# Chargement
df = load_kaggle_data()

if df is not None:
    # Filtres pour tester
    years = df['Year'].unique()
    selected_year = st.selectbox("Sélectionner une année pour vérifier", years, index=len(years)-1)
    
    # Affichage
    season_data = df[df['Year'] == selected_year]
    st.write(f"Données pour la saison {selected_year} ({len(season_data)} lignes)")
    st.dataframe(season_data, use_container_width=True)
    
    # Test Coéquipiers (pour vérifier que le merge Team a marché)
    st.subheader("Test Duels (Dernier GP)")
    last_round = season_data['Round'].max()
    last_gp = season_data[season_data['Round'] == last_round]
    
    for team in last_gp['Team'].unique():
        drivers = last_gp[last_gp['Team'] == team]['Surname'].tolist()
        st.write(f"**{team}:** {' vs '.join(drivers)}")