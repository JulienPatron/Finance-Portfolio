import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. LE MOTEUR MATHÉMATIQUE (CLASS) ---
class F1EloRating:
    def __init__(self, k_team_max=32, k_field=5):
        self.k_team_max = k_team_max 
        self.k_field = k_field       

    def _get_probability(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _get_actual_score(self, pos_a, pos_b):
        if pos_a < pos_b: return 1.0 
        elif pos_a > pos_b: return 0.0 
        return 0.5 

    def calculate_update(self, driver_rating, driver_pos, teammates_data, field_data):
        total_delta = 0.0

        # A. DUEL COÉQUIPIER
        nb_teammates = len(teammates_data)
        if nb_teammates > 0:
            k_team_adjusted = self.k_team_max / nb_teammates
            for mate in teammates_data:
                prob = self._get_probability(driver_rating, mate['rating'])
                actual = self._get_actual_score(driver_pos, mate['pos'])
                total_delta += (actual - prob) * k_team_adjusted

        # B. DUEL FIELD
        expected_score_sum = 0
        actual_score_sum = 0
        valid_opponents = 0
        
        for opponent in field_data:
            prob = self._get_probability(driver_rating, opponent['rating'])
            actual = self._get_actual_score(driver_pos, opponent['pos'])
            expected_score_sum += prob
            actual_score_sum += actual
            valid_opponents += 1
            
        if valid_opponents > 0:
            delta_field = (actual_score_sum - expected_score_sum) * self.k_field
            total_delta += delta_field

        return total_delta

# --- 2. CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_and_clean_data():
    try:
        races = pd.read_csv("data/races.csv")
        results = pd.read_csv("data/results.csv")
        drivers = pd.read_csv("data/drivers.csv")
        constructors = pd.read_csv("data/constructors.csv")
        
        df = pd.merge(results, races[['raceId', 'year', 'round', 'name', 'date']], on='raceId', how='left')
        df = pd.merge(df, drivers[['driverId', 'code', 'forename', 'surname']], on='driverId', how='left')
        df = pd.merge(df, constructors[['constructorId', 'name']], on='constructorId', how='left')
        
        df = df.rename(columns={'year': 'Year', 'round': 'Round', 'name_x': 'GP_Name', 'name_y': 'Team', 'positionOrder': 'Position', 'date': 'Date'})
        df['Full_Name'] = df['forename'] + " " + df['surname']
        
        df = df.sort_values(by=['Year', 'Round', 'Position'])
        # Keep Best Result
        df = df.drop_duplicates(subset=['Year', 'Round', 'driverId'], keep='first')
        
        return df[['Year', 'Round', 'Date', 'GP_Name', 'driverId', 'Full_Name', 'Team', 'Position']]
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None

# --- 3. BOUCLE DE CALCUL ---
@st.cache_data
def compute_elo_history(df):
    engine = F1EloRating()
    current_ratings = {}
    history_records = []
    
    races = df.groupby(['Year', 'Round', 'Date', 'GP_Name'], sort=False)
    progress_bar = st.progress(0)
    total_races = len(races)
    
    for i, ((year, round_num, date, gp), race_df) in enumerate(races):
        if i % 50 == 0: progress_bar.progress(i / total_races)
        
        race_drivers = []
        for _, row in race_df.iterrows():
            d_id = row['driverId']
            if d_id not in current_ratings:
                current_ratings[d_id] = 1500.0
            
            race_drivers.append({
                'id': d_id, 'name': row['Full_Name'], 'team': row['Team'],
                'pos': row['Position'], 'rating': current_ratings[d_id]
            })
            
        updates = {}
        for driver in race_drivers:
            teammates = [d for d in race_drivers if d['team'] == driver['team'] and d['id'] != driver['id']]
            field = [d for d in race_drivers if d['team'] != driver['team']]
            updates[driver['id']] = engine.calculate_update(driver['rating'], driver['pos'], teammates, field)
            
        for driver in race_drivers:
            new_rating = driver['rating'] + updates[driver['id']]
            current_ratings[driver['id']] = new_rating
            
            history_records.append({
                'Date': pd.to_datetime(date), 'Year': year, 'Driver': driver['name'],
                'Elo': new_rating, 'Team': driver['team']
            })
            
    progress_bar.empty()
    return pd.DataFrame(history_records)

# --- 4. INTERFACE STREAMLIT ---
st.set_page_config(page_title="F1 Elo Project", layout="wide")
st.title("🏆 F1 Elo Rating System")

df_raw = load_and_clean_data()

if df_raw is not None:
    with st.spinner("Calcul de l'histoire de la F1 en cours..."):
        df_elo = compute_elo_history(df_raw)
    
    # CRÉATION DES ONGLETS
    tab_analysis, tab_hof, tab_goat = st.tabs([
        "📉 Analyse Saison & Carrière", 
        "🏛️ Hall of Fame (All-Time)",
        "👑 Évolution du Record (GOATs)"
    ])
    
    # --- ONGLET 1 : ANALYSE ---
    with tab_analysis:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader("🔍 Filtres")
            years_list = sorted(df_elo['Year'].unique(), reverse=True)
            selected_year = st.selectbox("Année à afficher", years_list)
            
            all_drivers = sorted(df_elo['Driver'].unique())
            default_selection = ["Michael Schumacher", "Lewis Hamilton", "Max Verstappen", "Ayrton Senna", "Alain Prost"]
            valid_defaults = [d for d in default_selection if d in all_drivers]
            selected_drivers = st.multiselect("Comparer les pilotes", all_drivers, default=valid_defaults)

        with col2:
            st.subheader("📈 Courbes de Performance")
            if selected_drivers:
                chart_data = df_elo[df_elo['Driver'].isin(selected_drivers)].copy()
                
                # Graphique principal
                fig = px.line(chart_data, x='Date', y='Elo', color='Driver', 
                              title="Comparaison des carrières", render_mode='svg')
                
                # Zoom dynamique (Pas de 0)
                min_y = chart_data['Elo'].min() - 50
                max_y = chart_data['Elo'].max() + 50
                fig.update_layout(yaxis_range=[min_y, max_y], height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sélectionnez des pilotes.")

        st.divider()
        st.subheader(f"📊 Classement Fin de Saison {selected_year}")
        data_year = df_elo[df_elo['Year'] == selected_year]
        last_date = data_year['Date'].max()
        final_rankings = data_year[data_year['Date'] == last_date].sort_values(by='Elo', ascending=False).copy()
        final_rankings['Elo'] = final_rankings['Elo'].astype(int) # Entiers
        final_rankings = final_rankings.reset_index(drop=True)
        final_rankings.index += 1
        st.dataframe(final_rankings[['Driver', 'Team', 'Elo']], use_container_width=True)

    # --- ONGLET 2 : HALL OF FAME ---
    with tab_hof:
        st.subheader("👑 Les Meilleurs Pics de Performance (All-Time)")
        st.markdown("Score Elo **maximum** atteint par chaque pilote au sommet de sa carrière.")
        idx = df_elo.groupby(['Driver'])['Elo'].idxmax()
        best_elo_df = df_elo.loc[idx].copy()
        best_elo_df = best_elo_df.sort_values(by='Elo', ascending=False).reset_index(drop=True)
        best_elo_df.index += 1
        best_elo_df['Elo_Max'] = best_elo_df['Elo'].astype(int)
        st.dataframe(best_elo_df[['Driver', 'Elo_Max', 'Year', 'Team']], use_container_width=True, height=800)

    # --- ONGLET 3 : LA COURBE DES GOATS ---
    with tab_goat:
        st.subheader("🚀 La course au 'Greatest of All Time'")
        st.markdown("""
        Ce graphique montre **l'évolution du record historique de points Elo**.
        Une nouvelle marche n'est franchie que lorsqu'un pilote bat le record absolu précédent.
        """)
        
        # 1. Algorithme de détection des records
        # On trie toute l'histoire chronologiquement
        df_sorted = df_elo.sort_values(by='Date')
        
        goat_records = []
        current_max_elo = 0
        
        for _, row in df_sorted.iterrows():
            if row['Elo'] > current_max_elo:
                current_max_elo = row['Elo']
                goat_records.append({
                    'Date': row['Date'],
                    'Driver': row['Driver'],
                    'Record_Elo': int(current_max_elo)
                })
        
        df_goat = pd.DataFrame(goat_records)
        
        # 2. Le Graphique en "Marches d'escalier"
        # line_shape='hv' crée l'effet d'escalier (la ligne reste plate jusqu'au nouveau record)
        fig_goat = px.line(
            df_goat, 
            x='Date', 
            y='Record_Elo', 
            color='Driver',
            line_shape='hv', 
            markers=True,
            title="Historique du Record Absolu (Elo Peak)",
            labels={'Record_Elo': 'Record Elo Absolu'}
        )
        
        # Configuration visuelle
        fig_goat.update_layout(
            height=600,
            yaxis_range=[1500, df_goat['Record_Elo'].max() + 50], # Zoom intelligent
            legend_title="Détenteur du Record"
        )
        # Augmenter la taille des points pour voir quand le record est battu
        fig_goat.update_traces(marker=dict(size=8))
        
        st.plotly_chart(fig_goat, use_container_width=True)
        
        # 3. Le Tableau des passations de pouvoir
        st.write("📜 **Historique des passations de pouvoir :**")
        st.dataframe(
            df_goat[['Date', 'Driver', 'Record_Elo']].sort_values(by='Date', ascending=False), 
            use_container_width=True
        )