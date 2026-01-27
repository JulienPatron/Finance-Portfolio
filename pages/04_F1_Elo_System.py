import streamlit as st
import pandas as pd
import plotly.express as px

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="F1 Elo Rating System", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 1. LE MOTEUR MATHÉMATIQUE ---
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

# --- 2. CHARGEMENT ---
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

# --- 3. CALCUL ELO ---
@st.cache_data(show_spinner=False)
def compute_elo_history(df):
    engine = F1EloRating()
    current_ratings = {}
    history_records = []
    
    races = df.groupby(['Year', 'Round', 'Date', 'GP_Name'], sort=False)
    
    prog_bar = st.sidebar.progress(0, text="Initialisation...")
    total_races = len(races)
    
    for i, ((year, round_num, date, gp), race_df) in enumerate(races):
        if i % 100 == 0: 
            prog_bar.progress(i / total_races, text=f"Calcul {year}...")
        
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

# --- 4. FONCTION GAP ---
def calculate_teammate_gaps(final_rankings):
    gap_data = []
    for _, row in final_rankings.iterrows():
        driver = row['Driver']
        team = row['Team']
        elo = row['Elo']
        teammates = final_rankings[(final_rankings['Team'] == team) & (final_rankings['Driver'] != driver)]
        gap = 0
        mate_name = "Aucun"
        if not teammates.empty:
            teammates = teammates.copy()
            teammates['diff_abs'] = (teammates['Elo'] - elo).abs()
            closest_mate = teammates.sort_values('diff_abs').iloc[0]
            mate_name = closest_mate['Driver']
            gap = elo - closest_mate['Elo']
        gap_data.append({'Driver': driver, 'Team': team, 'Elo': elo, 'Gap': gap, 'Vs_Mate': mate_name})
    return pd.DataFrame(gap_data).sort_values(by='Gap', ascending=False)

# --- 5. INTERFACE ---

# === CSS PERSONNALISÉ (DESIGN) ===
st.markdown("""
<style>
    /* 0. REMONTER LE TITRE */
    .block-container {
        padding-top: 2rem;
    }

    /* 1. ONGLETS (TABS) */
    div[data-baseweb="tab-list"] {
        gap: 10px;
        margin-bottom: 20px;
    }
    div[data-baseweb="tab-highlight"] {
        display: none;
    }
    button[data-baseweb="tab"] {
        font-size: 20px !important;
        font-weight: 700 !important;
        padding: 10px 30px !important;
        background-color: #f0f2f6;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff !important;
        border-color: #ff4b4b !important;
        color: #ff4b4b !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* 2. SELECTEUR DE PILOTES (Make it Clean) */
    
    /* Le conteneur des tags : FOND BLANC */
    div[data-baseweb="select"] > div {
        background-color: #ffffff; 
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }

    /* Les Tags : NOUVEAU BLEU */
    span[data-baseweb="tag"] {
        background-color: #DBE6F7 !important; /* Le nouveau bleu demandé */
        color: #0c5460 !important;
        border: 1px solid #bee5eb;
        border-radius: 20px !important;
        padding: 2px 10px !important;
        font-size: 14px !important;
        margin-top: 2px !important;
        margin-bottom: 2px !important;
    }

    /* 3. METRICS */
    div[data-testid="stMetricValue"] { 
        font-size: 26px; 
    }
</style>
""", unsafe_allow_html=True)

st.title("F1 Elo Rating System")

df_raw = load_and_clean_data()

if df_raw is not None:
    if 'elo_data' not in st.session_state:
        st.session_state['elo_data'] = compute_elo_history(df_raw)
    df_elo = st.session_state['elo_data']

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Methode de Calcul")
        st.info("""
        **Principe Elo :**
        Chaque pilote commence a 1500 points.
        Apres chaque course, des points sont echanges en fonction des performances.
        """)
        st.markdown("""
        **1. Facteur Coequipier**
        Grosse ponderation. Le score evolue fortement selon la position d'arrivee par rapport au coequipier.
        
        **2. Facteur Field**
        Ponderation plus faible. Cela permet de situer le niveau global de la voiture (les bons pilotes ont souvent les bonnes voitures).
        """)

    # --- TABS ---
    tab_all_time, tab_season = st.tabs(["All Time", "Par Saison"])

    # --- TAB 1 : ALL TIME ---
    with tab_all_time:
        st.subheader("Comparateur de Pilotes")
        
        # On ajuste les proportions pour donner de l'air
        col_graph, col_select = st.columns([3.5, 1])
        
        with col_select:
            # Titre sans Emoji
            st.markdown("##### Ajouter des pilotes")
            
            all_drivers = sorted(df_elo['Driver'].unique())
            default_selection = ["Michael Schumacher", "Lewis Hamilton", "Max Verstappen", "Ayrton Senna", "Juan Manuel Fangio"]
            valid_defaults = [d for d in default_selection if d in all_drivers]
            
            selected_drivers = st.multiselect(
                "Select Drivers", 
                all_drivers, 
                default=valid_defaults, 
                label_visibility="collapsed"
            )
            
        with col_graph:
            if selected_drivers:
                chart_data = df_elo[df_elo['Driver'].isin(selected_drivers)].copy()
                fig = px.line(chart_data, x='Date', y='Elo', color='Driver', 
                              color_discrete_sequence=px.colors.qualitative.Bold)
                
                fig.update_layout(
                    height=500, 
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis_range=[chart_data['Elo'].min() - 50, chart_data['Elo'].max() + 50],
                    showlegend=True, 
                    legend=dict(orientation="h", y=-0.15, x=0)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Selectionnez des pilotes.")

        st.divider()
        
        col_goat, col_peak = st.columns(2)

        with col_goat:
            st.subheader("L'Histoire du Record")
            df_sorted = df_elo.sort_values(by='Date')
            goat_records = []
            current_max = 0
            for _, row in df_sorted.iterrows():
                if row['Elo'] > current_max:
                    current_max = row['Elo']
                    goat_records.append(row)
            df_goat = pd.DataFrame(goat_records)
            
            legend_order = df_goat.sort_values(by='Elo', ascending=False)['Driver'].unique().tolist()
            
            fig_goat = px.scatter(
                df_goat, x='Date', y='Elo', color='Driver',
                category_orders={"Driver": legend_order}
            )
            fig_goat.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
            fig_goat.update_layout(
                height=450, margin=dict(l=10, r=10, t=30, b=10),
                yaxis_range=[1500, df_goat['Elo'].max() + 50],
                showlegend=True, legend_title="Record Holders",
                yaxis_title="Record Elo Absolu"
            )
            st.plotly_chart(fig_goat, use_container_width=True)

        with col_peak:
            st.subheader("Les Plus Hauts Pics")
            idx = df_elo.groupby(['Driver'])['Elo'].idxmax()
            best_elo_df = df_elo.loc[idx].sort_values(by='Elo', ascending=False).reset_index(drop=True)
            best_elo_df.index += 1
            
            st.dataframe(
                best_elo_df[['Driver', 'Elo', 'Year', 'Team']],
                use_container_width=True,
                column_config={
                    "Elo": st.column_config.ProgressColumn("Peak Elo", format="%d", min_value=1500, max_value=2600),
                    "Year": st.column_config.NumberColumn("Annee", format="%d")
                }, height=450
            )

    # --- TAB 2 : PAR SAISON ---
    with tab_season:
        years_list = sorted(df_elo['Year'].unique(), reverse=True)
        col_sel, col_kpi = st.columns([1, 3])
        with col_sel:
            selected_year = st.selectbox("Choisir la saison", years_list)
        
        data_year = df_elo[df_elo['Year'] == selected_year].copy()
        last_date = data_year['Date'].max()
        final_rankings = data_year[data_year['Date'] == last_date].sort_values(by='Elo', ascending=False)
        champion = final_rankings.iloc[0]
        
        gap_df = calculate_teammate_gaps(final_rankings)
        champion_stats = gap_df[gap_df['Driver'] == champion['Driver']].iloc[0]
        
        with col_kpi:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Champion Elo", champion['Driver'])
            c2.metric("Score Final", f"{int(champion['Elo'])}")
            c3.metric("Ecurie", champion['Team'])
            
            gap_val = champion_stats['Gap']
            sign = "+" if gap_val > 0 else ""
            mate_name = champion_stats['Vs_Mate']
            c4.metric(f"Ecart vs {mate_name}", f"{sign}{int(gap_val)} pts")

        st.subheader(f"Progression sur la saison {selected_year}")
        sorted_drivers_legend = final_rankings['Driver'].tolist()
        top_10_drivers = final_rankings.head(10)['Driver'].tolist()
        chart_data_season = data_year[data_year['Driver'].isin(top_10_drivers)].copy()
        
        fig_season = px.line(
            chart_data_season, x='Date', y='Elo', color='Driver',
            markers=True, category_orders={"Driver": sorted_drivers_legend}
        )
        fig_season.update_layout(
            height=500, yaxis_range=[chart_data_season['Elo'].min() - 20, chart_data_season['Elo'].max() + 20],
            hovermode="x unified"
        )
        st.plotly_chart(fig_season, use_container_width=True)
        
        st.subheader("Domination Interne")
        st.dataframe(
            gap_df[['Driver', 'Team', 'Elo', 'Gap', 'Vs_Mate']].reset_index(drop=True),
            use_container_width=True,
            column_config={
                "Elo": st.column_config.NumberColumn("Elo", format="%d"),
                "Gap": st.column_config.NumberColumn("Ecart", format="%+d"),
                "Vs_Mate": "Compare a"
            }, height=600
        )

else:
    st.warning("Chargement des donnees...")