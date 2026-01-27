import streamlit as st
import pandas as pd
import plotly.express as px

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="F1 Elo Analytics", 
    layout="wide", 
    page_icon="🏎️",
    initial_sidebar_state="expanded"
)

# --- 1. LE MOTEUR MATHÉMATIQUE (IDENTIQUE) ---
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
        nb_teammates = len(teammates_data)
        if nb_teammates > 0:
            k_team_adjusted = self.k_team_max / nb_teammates
            for mate in teammates_data:
                prob = self._get_probability(driver_rating, mate['rating'])
                actual = self._get_actual_score(driver_pos, mate['pos'])
                total_delta += (actual - prob) * k_team_adjusted

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

# --- 2. CHARGEMENT (IDENTIQUE) ---
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
        df = df.drop_duplicates(subset=['Year', 'Round', 'driverId'], keep='first')
        
        return df[['Year', 'Round', 'Date', 'GP_Name', 'driverId', 'Full_Name', 'Team', 'Position']]
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None

# --- 3. BOUCLE DE CALCUL (IDENTIQUE) ---
@st.cache_data(show_spinner=False)
def compute_elo_history(df):
    engine = F1EloRating()
    current_ratings = {}
    history_records = []
    
    races = df.groupby(['Year', 'Round', 'Date', 'GP_Name'], sort=False)
    
    # Barre de chargement
    prog_bar = st.sidebar.progress(0, text="Initialisation du moteur Elo...")
    total_races = len(races)
    
    for i, ((year, round_num, date, gp), race_df) in enumerate(races):
        if i % 100 == 0: 
            prog_bar.progress(i / total_races, text=f"Calcul de la saison {year}...")
        
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
            
    prog_bar.empty()
    return pd.DataFrame(history_records)

# --- 4. INTERFACE UI/UX ---

st.title("🏎️ F1 Elo Analytics")
st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 24px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { font-weight: bold; }
</style>
""", unsafe_allow_html=True)

df_raw = load_and_clean_data()

if df_raw is not None:
    if 'elo_data' not in st.session_state:
        st.session_state['elo_data'] = compute_elo_history(df_raw)
    
    df_elo = st.session_state['elo_data']

    # --- SIDEBAR (SÉLECTION COMPARATEUR) ---
    with st.sidebar:
        st.header("⚙️ Paramètres")
        st.write("Choisissez les pilotes à comparer dans l'onglet 'All Time'.")
        
        all_drivers = sorted(df_elo['Driver'].unique())
        default_selection = ["Michael Schumacher", "Lewis Hamilton", "Max Verstappen", "Ayrton Senna", "Alain Prost", "Juan Manuel Fangio"]
        valid_defaults = [d for d in default_selection if d in all_drivers]
        
        selected_drivers = st.multiselect("Pilotes à comparer", all_drivers, default=valid_defaults)

    # --- LES DEUX ONGLETS PRINCIPAUX ---
    tab_all_time, tab_season = st.tabs(["🏛️ All Time & Légendes", "📅 Analyse par Saison"])

    # =========================================================================
    # ONGLET 1 : ALL TIME (Duel + GOAT + Peaks)
    # =========================================================================
    with tab_all_time:
        st.write("") # Spacer
        
        # --- SECTION HAUTE : 2 COLONNES (DUEL vs RECORDS) ---
        col_left, col_right = st.columns(2)
        
        # COLONNE GAUCHE : DUEL AU SOMMET
        with col_left:
            st.subheader("⚔️ Duel au Sommet")
            if selected_drivers:
                chart_data = df_elo[df_elo['Driver'].isin(selected_drivers)].copy()
                fig = px.line(chart_data, x='Date', y='Elo', color='Driver', 
                              color_discrete_sequence=px.colors.qualitative.Bold)
                
                # Zoom intelligent
                min_y = chart_data['Elo'].min() - 50
                max_y = chart_data['Elo'].max() + 50
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=30, b=10),
                    yaxis_range=[min_y, max_y],
                    showlegend=True,
                    legend=dict(orientation="h", y=1.1, x=0)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sélectionnez des pilotes dans la barre latérale.")

        # COLONNE DROITE : COURSE AU RECORD (GOAT)
        with col_right:
            st.subheader("⛰️ L'Histoire du Record (GOAT)")
            
            # Calcul dynamique du record
            df_sorted = df_elo.sort_values(by='Date')
            goat_records = []
            current_max = 0
            for _, row in df_sorted.iterrows():
                if row['Elo'] > current_max:
                    current_max = row['Elo']
                    goat_records.append(row)
            df_goat = pd.DataFrame(goat_records)
            
            fig_goat = px.line(df_goat, x='Date', y='Elo', color='Driver', 
                               line_shape='hv', markers=True) # hv = Escalier
            
            fig_goat.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis_range=[1500, df_goat['Elo'].max() + 50],
                showlegend=True,
                legend=dict(orientation="h", y=1.1, x=0),
                yaxis_title="Record Elo Absolu"
            )
            st.plotly_chart(fig_goat, use_container_width=True)

        st.divider()

        # --- SECTION BASSE : TABLEAU DES PICS ---
        st.subheader("🏆 Les Plus Hauts Pics de Carrière (Peak Elo)")
        
        idx = df_elo.groupby(['Driver'])['Elo'].idxmax()
        best_elo_df = df_elo.loc[idx].sort_values(by='Elo', ascending=False).reset_index(drop=True)
        best_elo_df.index += 1
        
        st.dataframe(
            best_elo_df[['Driver', 'Elo', 'Year', 'Team']],
            use_container_width=True,
            column_config={
                "Elo": st.column_config.ProgressColumn(
                    "Peak Elo",
                    format="%d",
                    min_value=1500,
                    max_value=2600,
                ),
                "Year": st.column_config.NumberColumn("Année du Pic", format="%d")
            },
            height=500
        )

    # =========================================================================
    # ONGLET 2 : ANALYSE SAISON (Progression)
    # =========================================================================
    with tab_season:
        # Sélecteur d'année spécifique à cet onglet
        years_list = sorted(df_elo['Year'].unique(), reverse=True)
        col_sel, col_kpi = st.columns([1, 3])
        
        with col_sel:
            selected_year = st.selectbox("📅 Choisir la saison à analyser", years_list)
        
        # Filtrer les données pour cette année
        data_year = df_elo[df_elo['Year'] == selected_year].copy()
        
        # KPIs
        last_date = data_year['Date'].max()
        final_rankings = data_year[data_year['Date'] == last_date].sort_values(by='Elo', ascending=False)
        top_driver = final_rankings.iloc[0]
        
        with col_kpi:
            c1, c2, c3 = st.columns(3)
            c1.metric("Champion Elo", top_driver['Driver'])
            c2.metric("Score Final", f"{int(top_driver['Elo'])}")
            c3.metric("Écurie", top_driver['Team'])

        st.subheader(f"📈 Progression sur la saison {selected_year}")
        
        # Pour ne pas avoir un graphique spaghetti avec 20 pilotes, 
        # on affiche automatiquement le Top 10 de cette année + les pilotes sélectionnés dans la sidebar
        top_10_drivers = final_rankings.head(10)['Driver'].tolist()
        drivers_to_plot = list(set(top_10_drivers + selected_drivers)) # Union sans doublons
        
        # Filtrer pour le graph
        chart_data_season = data_year[data_year['Driver'].isin(drivers_to_plot)].copy()
        
        fig_season = px.line(chart_data_season, x='Date', y='Elo', color='Driver',
                             title=f"Bataille pour le titre {selected_year} (Top 10 + Sélection)",
                             markers=True)
        
        # Zoom sur l'année
        min_y_s = chart_data_season['Elo'].min() - 20
        max_y_s = chart_data_season['Elo'].max() + 20
        
        fig_season.update_layout(
            height=500,
            yaxis_range=[min_y_s, max_y_s],
            hovermode="x unified"
        )
        st.plotly_chart(fig_season, use_container_width=True)
        
        st.subheader("Classement Final")
        st.dataframe(
            final_rankings[['Driver', 'Team', 'Elo']].reset_index(drop=True),
            use_container_width=True,
            column_config={
                "Elo": st.column_config.ProgressColumn(
                    "Puissance Elo",
                    format="%d",
                    min_value=1400,
                    max_value=2600,
                ),
            }
        )

else:
    st.warning("Chargement des données...")