import streamlit as st
import pandas as pd
import os

# Note : Pas de set_page_config, géré par main.py

# ==============================================================================
# 1. MOTEUR MATHÉMATIQUE (ELO)
# ==============================================================================

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
        
        # 1. Teammate Battle (Poids Fort)
        nb_teammates = len(teammates_data)
        if nb_teammates > 0:
            k_team_adjusted = self.k_team_max / nb_teammates
            for mate in teammates_data:
                prob = self._get_probability(driver_rating, mate['rating'])
                actual = self._get_actual_score(driver_pos, mate['pos'])
                total_delta += (actual - prob) * k_team_adjusted

        # 2. Field Battle (Poids Faible)
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

# ==============================================================================
# 2. CHARGEMENT DES DONNÉES (CACHE & OPTIMISÉ)
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_and_clean_data():
    try:
        # Gestion intelligente du chemin (root vs pages dir)
        base_dir = "data"
        if not os.path.exists(base_dir):
            base_dir = "../data" # Fallback si exécuté depuis un sous-dossier
            
        # Chargement optimisé avec types spécifiques
        races = pd.read_csv(f"{base_dir}/races.csv")
        results = pd.read_csv(f"{base_dir}/results.csv")
        drivers = pd.read_csv(f"{base_dir}/drivers.csv")
        constructors = pd.read_csv(f"{base_dir}/constructors.csv")
        
        # Merge
        df = pd.merge(results, races[['raceId', 'year', 'round', 'name', 'date']], on='raceId', how='left')
        df = pd.merge(df, drivers[['driverId', 'code', 'forename', 'surname']], on='driverId', how='left')
        df = pd.merge(df, constructors[['constructorId', 'name']], on='constructorId', how='left')
        
        # Renommage et Création colonnes
        df = df.rename(columns={'year': 'Year', 'round': 'Round', 'name_x': 'GP_Name', 'name_y': 'Team', 'positionOrder': 'Position', 'date': 'Date'})
        df['Full_Name'] = df['forename'] + " " + df['surname']
        
        # Conversion Date (Optimisation temps)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Nettoyage
        df = df.sort_values(by=['Year', 'Round', 'Position'])
        df = df.drop_duplicates(subset=['Year', 'Round', 'driverId'], keep='first')
        
        return df[['Year', 'Round', 'Date', 'GP_Name', 'driverId', 'Full_Name', 'Team', 'Position']]
    except Exception as e:
        st.error(f"Erreur de chargement des données CSV : {e}")
        return None

# ==============================================================================
# 3. CALCUL DE L'HISTORIQUE ELO (LOURD - CACHÉ)
# ==============================================================================

@st.cache_data(show_spinner=False)
def compute_elo_history(df):
    engine = F1EloRating()
    current_ratings = {}
    history_records = []
    
    # GroupBy préserve l'ordre chronologique si le DF est trié
    races = df.groupby(['Year', 'Round', 'Date', 'GP_Name'], sort=False)
    
    # Barre de progression
    prog_bar = st.sidebar.progress(0, text="Initialisation du moteur Elo...")
    total_races = len(races)
    
    for i, ((year, round_num, date, gp), race_df) in enumerate(races):
        # Update UI moins fréquent pour gagner en perf
        if i % 50 == 0: 
            prog_bar.progress(i / total_races, text=f"Calcul de la saison {year}...")
        
        # Préparation des structures de données pour la course
        race_drivers = []
        for _, row in race_df.iterrows():
            d_id = row['driverId']
            if d_id not in current_ratings:
                current_ratings[d_id] = 1500.0
            
            race_drivers.append({
                'id': d_id, 'name': row['Full_Name'], 'team': row['Team'],
                'pos': row['Position'], 'rating': current_ratings[d_id]
            })
            
        # Calcul des deltas
        updates = {}
        for driver in race_drivers:
            # Filtrage des coéquipiers vs reste du plateau
            teammates = [d for d in race_drivers if d['team'] == driver['team'] and d['id'] != driver['id']]
            field = [d for d in race_drivers if d['team'] != driver['team']]
            
            updates[driver['id']] = engine.calculate_update(driver['rating'], driver['pos'], teammates, field)
            
        # Application des mises à jour
        for driver in race_drivers:
            new_rating = driver['rating'] + updates[driver['id']]
            current_ratings[driver['id']] = new_rating
            
            # On stocke l'historique
            history_records.append({
                'Date': date, 'Year': year, 'Driver': driver['name'],
                'Elo': new_rating, 'Team': driver['team']
            })
            
    prog_bar.empty()
    return pd.DataFrame(history_records)

def calculate_teammate_gaps(final_rankings):
    gap_data = []
    # Optimisation vectorielle difficile ici, on garde la boucle
    for _, row in final_rankings.iterrows():
        driver = row['Driver']
        team = row['Team']
        elo = row['Elo']
        
        # Trouver le coéquipier
        teammates = final_rankings[(final_rankings['Team'] == team) & (final_rankings['Driver'] != driver)]
        
        gap = 0
        mate_name = "None"
        
        if not teammates.empty:
            # On prend le coéquipier le plus proche en terme d'Elo (cas où il y a 3 pilotes)
            # ou simplement l'autre pilote
            teammates = teammates.copy()
            teammates['diff_abs'] = (teammates['Elo'] - elo).abs()
            closest_mate = teammates.sort_values('diff_abs').iloc[0]
            
            mate_name = closest_mate['Driver']
            gap = elo - closest_mate['Elo']
        
        gap_data.append({'Driver': driver, 'Team': team, 'Elo': elo, 'Gap': gap, 'Vs_Mate': mate_name})
        
    return pd.DataFrame(gap_data).sort_values(by='Gap', ascending=False)

# ==============================================================================
# 4. INTERFACE UTILISATEUR
# ==============================================================================

# CSS Optimisé
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    /* Tabs Style */
    button[data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    /* Driver Tags */
    span[data-baseweb="tag"] {
        background-color: #DBE6F7 !important; 
        color: #095AA7 !important;
        border: 1px solid #bee5eb;
    }
    /* Metrics */
    div[data-testid="stMetricValue"] { font-size: 24px; }
</style>
""", unsafe_allow_html=True)

st.title("F1 Elo Rating System")

with st.spinner("Chargement de l'historique F1..."):
    df_raw = load_and_clean_data()

if df_raw is not None:
    # Calcul ou récupération du cache Elo
    if 'elo_data' not in st.session_state:
        st.session_state['elo_data'] = compute_elo_history(df_raw)
    df_elo = st.session_state['elo_data']

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Méthodologie")
        st.info("""
        **Principe Elo :** Chaque pilote démarre à 1500.
        **Facteurs Clés :**
        1. **Duel Coéquipier (High K):** Performance relative à la voiture.
        2. **Position Piste (Low K):** Performance globale.
        """)

    # --- TABS ---
    tab_all_time, tab_season = st.tabs(["📈 Historique Global", "📅 Par Saison"])

    # --- TAB 1 : ALL TIME ---
    with tab_all_time:
        st.subheader("Comparateur de Pilotes")
        
        col_graph, col_select = st.columns([3.5, 1])
        
        with col_select:
            all_drivers = sorted(df_elo['Driver'].unique())
            default_selection = ["Michael Schumacher", "Lewis Hamilton", "Max Verstappen", "Ayrton Senna", "Alain Prost"]
            valid_defaults = [d for d in default_selection if d in all_drivers]
            
            selected_drivers = st.multiselect("Sélectionner pilotes", all_drivers, default=valid_defaults)
            
        with col_graph:
            if selected_drivers:
                # Import Lazy de Plotly
                import plotly.express as px
                
                chart_data = df_elo[df_elo['Driver'].isin(selected_drivers)].copy()
                fig = px.line(chart_data, x='Date', y='Elo', color='Driver', 
                              color_discrete_sequence=px.colors.qualitative.Bold)
                
                fig.update_layout(
                    height=450, 
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis_range=[chart_data['Elo'].min() - 50, chart_data['Elo'].max() + 50],
                    legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sélectionnez des pilotes pour voir le graphique.")

        st.divider()
        
        col_goat, col_peak = st.columns(2)

        with col_goat:
            st.subheader("Historique du Record Absolu")
            # Calcul du recordman au fil du temps
            df_sorted = df_elo.sort_values(by='Date')
            goat_records = []
            current_max = 0
            
            for _, row in df_sorted.iterrows():
                if row['Elo'] > current_max:
                    current_max = row['Elo']
                    goat_records.append(row)
            
            df_goat = pd.DataFrame(goat_records)
            
            import plotly.express as px # Lazy Import
            
            fig_goat = px.scatter(
                df_goat, x='Date', y='Elo', color='Driver',
                category_orders={"Driver": df_goat['Driver'].unique().tolist()} # Ordre d'apparition
            )
            fig_goat.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
            fig_goat.update_layout(
                height=400, margin=dict(t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_goat, use_container_width=True)

        with col_peak:
            st.subheader("Top Peaks (Elo Max)")
            idx = df_elo.groupby(['Driver'])['Elo'].idxmax()
            best_elo_df = df_elo.loc[idx].sort_values(by='Elo', ascending=False).head(20).reset_index(drop=True)
            best_elo_df.index += 1
            
            st.dataframe(
                best_elo_df[['Driver', 'Elo', 'Year', 'Team']],
                use_container_width=True,
                column_config={
                    "Elo": st.column_config.ProgressColumn("Peak Elo", format="%d", min_value=1500, max_value=2600),
                    "Year": st.column_config.NumberColumn("Year", format="%d")
                }, 
                height=400
            )

    # --- TAB 2 : BY SEASON ---
    with tab_season:
        years_list = sorted(df_elo['Year'].unique(), reverse=True)
        col_sel, col_kpi = st.columns([1, 3])
        
        with col_sel:
            selected_year = st.selectbox("Saison", years_list)
        
        # Filtrage données
        data_year = df_elo[df_elo['Year'] == selected_year].copy()
        
        # Classement Final de la saison
        last_date = data_year['Date'].max()
        final_rankings = data_year[data_year['Date'] == last_date].sort_values(by='Elo', ascending=False)
        champion = final_rankings.iloc[0]
        
        # Gap analysis
        gap_df = calculate_teammate_gaps(final_rankings)
        champion_stats = gap_df[gap_df['Driver'] == champion['Driver']].iloc[0]
        
        with col_kpi:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Champion (Elo)", champion['Driver'])
            c2.metric("Score Final", f"{int(champion['Elo'])}")
            c3.metric("Écurie", champion['Team'])
            
            gap_val = champion_stats['Gap']
            sign = "+" if gap_val > 0 else ""
            c4.metric(f"Gap vs {champion_stats['Vs_Mate']}", f"{sign}{int(gap_val)} pts")

        # Graphique Saison
        st.subheader(f"Progression Saison {selected_year}")
        
        top_10_drivers = final_rankings.head(10)['Driver'].tolist()
        chart_data_season = data_year[data_year['Driver'].isin(top_10_drivers)].copy()
        
        import plotly.express as px # Lazy Import
        
        fig_season = px.line(
            chart_data_season, x='Date', y='Elo', color='Driver',
            markers=True, category_orders={"Driver": top_10_drivers}
        )
        fig_season.update_layout(
            height=450, 
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_season, use_container_width=True)
        
        # Tableau Domination Interne
        st.subheader("Domination Interne (Pilote vs Coéquipier)")
        st.dataframe(
            gap_df[['Driver', 'Team', 'Elo', 'Gap', 'Vs_Mate']].reset_index(drop=True),
            use_container_width=True,
            column_config={
                "Elo": st.column_config.NumberColumn("Elo", format="%d"),
                "Gap": st.column_config.NumberColumn("Gap vs Mate", format="%+d"),
                "Vs_Mate": "Comparé à"
            },
            height=500
        )

else:
    st.warning("Impossible de charger les données CSV. Vérifiez le dossier 'data/'.")