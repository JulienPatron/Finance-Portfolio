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

# --- 4. FONCTION UTILITAIRE : CALCUL DES ÉCARTS COÉQUIPIERS ---
def calculate_teammate_gaps(final_rankings):
    """
    Crée un DataFrame avec l'écart Elo vis-à-vis du coéquipier le plus proche.
    """
    gap_data = []
    
    for _, row in final_rankings.iterrows():
        driver = row['Driver']
        team = row['Team']
        elo = row['Elo']
        
        # Trouver les coéquipiers dans la même écurie
        teammates = final_rankings[
            (final_rankings['Team'] == team) & 
            (final_rankings['Driver'] != driver)
        ]
        
        gap = 0
        mate_name = "Aucun"
        
        if not teammates.empty:
            # On cherche le coéquipier le plus proche en niveau Elo (valeur absolue)
            teammates = teammates.copy()
            teammates['diff_abs'] = (teammates['Elo'] - elo).abs()
            closest_mate = teammates.sort_values('diff_abs').iloc[0]
            
            mate_name = closest_mate['Driver']
            gap = elo - closest_mate['Elo'] # Positif = Je suis meilleur
            
        gap_data.append({
            'Driver': driver,
            'Team': team,
            'Elo': elo,
            'Gap': gap,
            'Vs_Mate': mate_name
        })
        
    return pd.DataFrame(gap_data).sort_values(by='Gap', ascending=False)

# --- 5. INTERFACE UI/UX ---

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

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Paramètres")
        st.write("Pilotes pour l'onglet 'All Time'.")
        all_drivers = sorted(df_elo['Driver'].unique())
        default_selection = ["Michael Schumacher", "Lewis Hamilton", "Max Verstappen", "Ayrton Senna", "Alain Prost", "Juan Manuel Fangio"]
        valid_defaults = [d for d in default_selection if d in all_drivers]
        selected_drivers = st.multiselect("Pilotes à comparer", all_drivers, default=valid_defaults)

    # --- TABS ---
    tab_all_time, tab_season = st.tabs(["🏛️ All Time & Légendes", "📅 Analyse par Saison"])

    # =========================================================================
    # ONGLET 1 : ALL TIME
    # =========================================================================
    with tab_all_time:
        st.write("") 
        col_left, col_right = st.columns(2)
        
        # GAUCHE : DUEL
        with col_left:
            st.subheader("⚔️ Duel au Sommet")
            if selected_drivers:
                chart_data = df_elo[df_elo['Driver'].isin(selected_drivers)].copy()
                fig = px.line(chart_data, x='Date', y='Elo', color='Driver', 
                              color_discrete_sequence=px.colors.qualitative.Bold)
                
                fig.update_layout(
                    height=450,
                    margin=dict(l=10, r=10, t=30, b=50),
                    yaxis_range=[chart_data['Elo'].min() - 50, chart_data['Elo'].max() + 50],
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.2, x=0) # Légende en bas
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sélectionnez des pilotes.")

        # DROITE : GOAT
        with col_right:
            st.subheader("⛰️ L'Histoire du Record (GOAT)")
            df_sorted = df_elo.sort_values(by='Date')
            goat_records = []
            current_max = 0
            for _, row in df_sorted.iterrows():
                if row['Elo'] > current_max:
                    current_max = row['Elo']
                    goat_records.append(row)
            df_goat = pd.DataFrame(goat_records)
            
            fig_goat = px.line(df_goat, x='Date', y='Elo', color='Driver', 
                               line_shape='hv', markers=True)
            
            fig_goat.update_layout(
                height=450,
                margin=dict(l=10, r=10, t=30, b=50),
                yaxis_range=[1500, df_goat['Elo'].max() + 50],
                showlegend=True,
                legend=dict(orientation="h", y=-0.2, x=0), # Légende en bas pour ne pas manger le graph
                yaxis_title="Record Elo Absolu"
            )
            st.plotly_chart(fig_goat, use_container_width=True)

        st.divider()
        st.subheader("🏆 Les Plus Hauts Pics de Carrière (Peak Elo)")
        
        idx = df_elo.groupby(['Driver'])['Elo'].idxmax()
        best_elo_df = df_elo.loc[idx].sort_values(by='Elo', ascending=False).reset_index(drop=True)
        best_elo_df.index += 1
        
        st.dataframe(
            best_elo_df[['Driver', 'Elo', 'Year', 'Team']],
            use_container_width=True,
            column_config={
                "Elo": st.column_config.ProgressColumn("Peak Elo", format="%d", min_value=1500, max_value=2600),
                "Year": st.column_config.NumberColumn("Année du Pic", format="%d")
            },
            height=500
        )

    # =========================================================================
    # ONGLET 2 : ANALYSE SAISON
    # =========================================================================
    with tab_season:
        years_list = sorted(df_elo['Year'].unique(), reverse=True)
        col_sel, col_kpi = st.columns([1, 3])
        
        with col_sel:
            selected_year = st.selectbox("📅 Choisir la saison", years_list)
        
        # Données de l'année
        data_year = df_elo[df_elo['Year'] == selected_year].copy()
        last_date = data_year['Date'].max()
        
        # Classement final brut
        final_rankings = data_year[data_year['Date'] == last_date].sort_values(by='Elo', ascending=False)
        
        # Calcul des écarts coéquipiers
        gap_df = calculate_teammate_gaps(final_rankings)
        top_driver = gap_df.iloc[0] # Le pilote avec le Elo le plus haut (car trié par Elo avant calcul) mais gap_df est trié par Gap...
        # Attention: gap_df est trié par GAP. Retrouvons le champion (plus haut Elo)
        champion = final_rankings.iloc[0]
        champion_stats = gap_df[gap_df['Driver'] == champion['Driver']].iloc[0]
        
        with col_kpi:
            c1, c2, c3 = st.columns(3)
            c1.metric("Champion Elo", champion['Driver'])
            c2.metric("Score Final", f"{int(champion['Elo'])}")
            
            gap_val = champion_stats['Gap']
            sign = "+" if gap_val > 0 else ""
            mate_txt = f"vs {champion_stats['Vs_Mate']}" if champion_stats['Vs_Mate'] != "Aucun" else "Sans coéquipier"
            c3.metric(f"Écart {mate_txt}", f"{sign}{int(gap_val)} pts")

        st.subheader(f"📈 Progression sur la saison {selected_year}")
        
        # Liste des pilotes triée par ordre Elo FINAL (pour la légende)
        sorted_drivers_legend = final_rankings['Driver'].tolist()
        
        # Top 10 + Sélection
        top_10_drivers = final_rankings.head(10)['Driver'].tolist()
        drivers_to_plot = list(set(top_10_drivers + selected_drivers))
        chart_data_season = data_year[data_year['Driver'].isin(drivers_to_plot)].copy()
        
        fig_season = px.line(
            chart_data_season, 
            x='Date', y='Elo', color='Driver',
            markers=True,
            # C'EST ICI QU'ON FORCE L'ORDRE DE LA LÉGENDE
            category_orders={"Driver": sorted_drivers_legend} 
        )
        
        fig_season.update_layout(
            height=500,
            yaxis_range=[chart_data_season['Elo'].min() - 20, chart_data_season['Elo'].max() + 20],
            hovermode="x unified"
        )
        st.plotly_chart(fig_season, use_container_width=True)
        
        st.subheader("⚔️ Domination Interne (Écart vs Coéquipier)")
        st.write("Ce tableau classe les pilotes par l'écart de points creusé face à leur coéquipier le plus proche.")
        
        st.dataframe(
            gap_df[['Driver', 'Team', 'Elo', 'Gap', 'Vs_Mate']].reset_index(drop=True),
            use_container_width=True,
            column_config={
                "Elo": st.column_config.NumberColumn("Elo", format="%d"),
                "Gap": st.column_config.NumberColumn(
                    "Écart", 
                    format="%+d", # Affiche le + ou le -
                    help="Différence de points avec le coéquipier le plus proche"
                ),
                "Vs_Mate": "Comparé à"
            },
            height=600
        )

else:
    st.warning("Chargement des données...")