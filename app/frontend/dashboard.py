import os
import sys
import pandas as pd
import streamlit as st
from pathlib import Path
import pulp

# Path magic to allow importing from src while running from app/frontend/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config
from src.inference.optimizer import generate_optimal_lineup
from src.models.predict import run_inference

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="WNBA MLOps Engine",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data(filepath):
    """Safely loads CSVs with Streamlit caching for performance."""
    if Path(filepath).exists():
        return pd.read_csv(filepath)
    return pd.DataFrame() # Return empty if not found yet

# --- MAIN UI ---
st.title("🏀 WNBA DFS MLOps Engine")
st.markdown("Automated predictive pipeline and lineup optimization.")

# Initialize the Tabs
tab_predict, tab_vault, tab_drift = st.tabs([
    "🔮 Daily Optimizer", 
    "🗄️ Data Vault & OSINT", 
    "📈 Model Health"
])


# ==========================================
# TAB 1: DAILY OPTIMIZER
# ==========================================
with tab_predict:
    st.header("🔮 Daily Optimizer")
    st.markdown("Upload the official DraftKings salary CSV to generate the mathematically optimal WNBA lineup.")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload DraftKings Salaries (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        # Load the raw DraftKings data
        dk_df = pd.read_csv(uploaded_file)
        
        # Ensure it's a valid DK file by checking for standard columns
        if 'Name' not in dk_df.columns or 'Salary' not in dk_df.columns or 'Position' not in dk_df.columns:
            st.error("Invalid CSV format. Please upload the raw DraftKings salary file.")
        else:
            with st.expander("🔍 View Raw Slate Data", expanded=False):
                st.dataframe(dk_df[['Name', 'Position', 'Salary', 'TeamAbbrev', 'Game Info']].head(10))
                
            st.divider()
            
            # Execution Button
            if st.button("🚀 Run Inference & Optimize Lineup", type="primary", use_container_width=True):
                
                with st.spinner("Fetching historical anchors, computing proxies, and running XGBoost Inference..."):
                    
                    # --- THE ACTUAL MLOPS INFERENCE HOOK ---
                    dk_df = run_inference(dk_df)
                    # ---------------------------------------
                    
                    # 3. Call the isolated Optimizer Engine
                    results = generate_optimal_lineup(dk_df)
                    
                    # 4. Render the Results
                    if results["status"] == "Optimal":
                        st.success("Mathematical Optimization Complete!")
                        
                        total_salary = results["total_salary"]
                        total_projection = results["total_points"]
                        optimal_lineup = results["lineup_df"]
                        
                        # Display Top-Level Metrics
                        col_sal, col_proj, col_value = st.columns(3)
                        col_sal.metric("Total Salary", f"${total_salary:,}", delta=f"${50000 - total_salary:,} Remainder", delta_color="off")
                        col_proj.metric("Projected Points", f"{total_projection:.2f}")
                        col_value.metric("Team Value (Pts/$)", f"{(total_projection / total_salary) * 1000:.2f}x")
                        
                        # Display Lineup Card
                        st.subheader("🏆 Optimal Starting 6")
                        
                        display_df = optimal_lineup[['Position', 'Name', 'Salary', 'TeamAbbrev', 'Predicted_Pts']]
                        display_df = display_df.sort_values(by=['Position', 'Salary'], ascending=[False, False]).reset_index(drop=True)
                        display_df['Salary'] = display_df['Salary'].apply(lambda x: f"${x:,}")
                        display_df['Predicted_Pts'] = display_df['Predicted_Pts'].apply(lambda x: f"{x:.1f}")
                        
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                    else:
                        st.error(results["status"])

# ==========================================
# TAB 2: DATA VAULT & OSINT
# ==========================================
with tab_vault:
    st.header("Pipeline Data Health")
    st.markdown("Monitor the state of the Open Source Intelligence (OSINT) proxies and historical data vault.")
    
    # Load the Data
    vault_df = load_data(config.ROSTERS_DATA_DIR / f"player_vault_{config.CURRENT_SEASON}.csv")
    proxies_df = load_data(config.PROCESSED_DATA_DIR / "rookie_proxies.csv")
    golden_df = load_data(config.PROCESSED_DATA_DIR / "training_features.csv")

    ncaa_count = len(proxies_df[proxies_df['ORIGIN_LEAGUE'] == 'NCAA']) if not proxies_df.empty else 0
    intl_count = len(proxies_df[proxies_df['ORIGIN_LEAGUE'] == 'INTL']) if not proxies_df.empty else 0

    # 1. Top-Level Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Players in Vault", len(vault_df) if not vault_df.empty else 0)
    col2.metric("NCAA Proxies Loaded", ncaa_count)
    col3.metric("INTL Proxies Loaded", intl_count)
    col4.metric("Total Historical Rows", f"{len(golden_df):,}" if not golden_df.empty else 0)

    st.divider()

    # 2. The Proxy Database
    st.subheader("Rookie Translation Proxies")
    col_ncaa, col_intl = st.columns(2)

    with col_ncaa:
        st.markdown(f"**NCAA Draft Targets ({config.NCAA_ROOKIE_TAX} Tax)**")
        ncaa_df = proxies_df[proxies_df['ORIGIN_LEAGUE'] == 'NCAA'] if not proxies_df.empty else pd.DataFrame()
        if not ncaa_df.empty:
            st.dataframe(ncaa_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No NCAA proxies found.")

    with col_intl:
        st.markdown(f"**International Pro Targets ({config.INTL_PRO_TAX} Tax)**")
        intl_df = proxies_df[proxies_df['ORIGIN_LEAGUE'] == 'INTL'] if not proxies_df.empty else pd.DataFrame()
        if not intl_df.empty:
            st.dataframe(intl_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No International proxies found.")

    st.divider()

    # 3. Golden Table Spotlight
    st.subheader("Feature Engineering (Golden Table)")
    st.markdown("A live view of the processed features currently fed to the XGBoost model.")
    if not golden_df.empty:
        # Show the most recent 100 rows, sorted by Date descending
        display_df = golden_df.sort_values(by='GAME_DATE', ascending=False).head(100)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ==========================================
# TAB 3: MODEL HEALTH (Placeholder)
# ==========================================
with tab_drift:
    st.header("Model Performance & Drift")
    st.info("Rolling Mean Absolute Error (MAE) and Backtest metrics will be tracked here.")