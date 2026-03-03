import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import pulp
import unicodedata
from pathlib import Path

# --- Path Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config

# ==========================================
# 1. UTILITY FUNCTIONS (The Cleaners)
# ==========================================

def normalize_name(name):
    """Standardizes names to link Box Scores with the API Vault."""
    if not isinstance(name, str): return ""
    name = name.lower().strip()
    # Remove accents
    name = "".join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    # Remove punctuation
    name = name.replace("'", "").replace("-", " ").replace(".", "")
    return name

def normalize_position(pos):
    """Converts WNBA API positions cleanly to DraftKings DFS slots."""
    pos = str(pos).upper().strip()
    
    # 1. Handle missing data safely to prevent pipeline crashes
    if pos in ['NAN', 'NONE', '']:
        return 'FORWARD' 
        
    # 2. Exorcise the ghost from the old CSVs
    pos = pos.replace('FENTER', 'FORWARD')
        
    # 3. Handle the full word FIRST before looking at single letters
    pos = pos.replace('CENTER', 'FORWARD')
    
    # 4. Standardize the abbreviations mapping Centers to Forwards
    if pos in ['C', 'F-C', 'C-F', 'F']:
        return 'FORWARD'
    elif pos == 'G':
        return 'GUARD'
        
    # 5. Leave combo strings intact (e.g., "GUARD-FORWARD") so the optimizer 
    # can trigger both the 'G' and 'F' eligibility checks!
    return pos

# ==========================================
# 2. THE OPTIMIZER (The Math Engine)
# ==========================================

def solve_lineup(players_df):
    """Runs Linear Programming to find the max-point lineup under Salary constraints."""
    prob = pulp.LpProblem("WNBA_Lineup_Optimization", pulp.LpMaximize)
    
    # Decision Variables: 0 or 1 for every player
    player_vars = pulp.LpVariable.dicts("Players", players_df.index, cat='Binary')
    
    # Objective: Maximize Predicted Points
    prob += pulp.lpSum([players_df.loc[i, 'predicted_pts'] * player_vars[i] for i in players_df.index])
    
    # Constraint 1: Salary Cap ($50,000)
    prob += pulp.lpSum([players_df.loc[i, 'salary'] * player_vars[i] for i in players_df.index]) <= config.SALARY_CAP
    
    # Constraint 2: Total Roster Size (6 players)
    prob += pulp.lpSum([player_vars[i] for i in players_df.index]) == config.TOTAL_SLOTS
    
    # Constraint 3: Position Minimums (Allows for Dual-Eligibility & Utility Slot)
    is_guard = players_df['POSITION'].str.contains('G', na=False)
    is_forward = players_df['POSITION'].str.contains('F', na=False)
    
    prob += pulp.lpSum([player_vars[i] for i in players_df.index if is_guard[i]]) >= config.ROSTER_SLOTS['G']
    prob += pulp.lpSum([player_vars[i] for i in players_df.index if is_forward[i]]) >= config.ROSTER_SLOTS['F']
    
    # Solve silently
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Extract Winners
    chosen_indices = [i for i in players_df.index if player_vars[i].varValue == 1]
    return players_df.loc[chosen_indices]

# ==========================================
# 3. THE MAIN BACKTEST SIMULATOR
# ==========================================

def run_backtest(target_date):
    print(f"🕵️  Initiating Gameday Simulation for: {target_date}")
    
    # --- A. Load Model & Features ---
    try:
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(project_root, "src", "models", "model.ubj"))
    except Exception as e:
        print(f"❌ Could not load XGBoost model. Did you run train.py? Error: {e}")
        return

    features_path = config.PROCESSED_DATA_DIR / "training_features.csv"
    df = pd.read_csv(features_path)
    
    # Filter for the specific day
    day_df = df[df['GAME_DATE'] == target_date].copy()
    if day_df.empty:
        print(f"⚠️ No games found in the dataset for {target_date}. Try another date!")
        return
        
    # --- B. Load & Normalize Vault ---
    vault_path = config.DATA_DIR / "metadata" / "player_vault_final.csv" # Or player_vault_2025.csv
    try:
        vault_df = pd.read_csv(vault_path)
    except FileNotFoundError:
        print("⚠️ Vault not found. Run the scraper/hydration scripts first!")
        return

    print(f"🔄 Merging {len(day_df)} game records with API Vault Metadata...")
    day_df['match_name'] = day_df['PLAYER_NAME'].apply(normalize_name)
    vault_df['match_name'] = vault_df['PLAYER_NAME'].apply(normalize_name)
    
    # Left join to ensure we don't lose players
    day_df = day_df.merge(vault_df[['match_name', 'POSITION']], on='match_name', how='left')
    day_df['POSITION'] = day_df['POSITION'].apply(normalize_position)

    # --- C. Generate Proxy Salaries & Predictions ---
    # DFS sites usually price players at ~250x their historical average
    day_df['salary'] = (day_df['FPTS_SEASON_AVG'] * 250).clip(lower=3500, upper=12000)
    
    # Format data for XGBoost (Drop metadata, keep exact columns used in training)
    to_drop = config.DROPPED_FEATURES + [config.TARGET_COL, 'match_name', 'POSITION', 'salary']
    actual_drops = [c for c in to_drop if c in day_df.columns]
    X_predict = day_df.drop(columns=actual_drops).select_dtypes(include=['number'])
    
    print("🧠 Generating XGBoost Predictions...")
    day_df['predicted_pts'] = model.predict(X_predict)

    # --- D. Run Optimizer ---
    print("🧮 Running Linear Programming Optimizer...")
    optimal_lineup = solve_lineup(day_df.reset_index(drop=True))

    # --- E. Reveal Results ---
    total_predicted = optimal_lineup['predicted_pts'].sum()
    total_actual = optimal_lineup[config.TARGET_COL].sum()
    total_salary = optimal_lineup['salary'].sum()

    print("\n" + "="*50)
    print(f"🏆 OPTIMAL LINEUP: {target_date}")
    print("="*50)
    
    # Format the printout nicely
    display_cols = ['PLAYER_NAME', 'POSITION', 'salary', 'predicted_pts', config.TARGET_COL]
    print(optimal_lineup[display_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    print("-" * 50)
    print(f"💰 Total Salary Used: ${total_salary:,.0f} / ${config.SALARY_CAP:,.0f}")
    print(f"🔮 Predicted Fantasy Points: {total_predicted:.2f}")
    print(f"🔥 ACTUAL Fantasy Points:   {total_actual:.2f}")
    
    # The ultimate ROI check
    if total_actual >= 150:
        print("✅ RESULT: CASH LINE CLEARED! This lineup likely won money.")
    else:
        print("❌ RESULT: MISSED CASH LINE. (Usually need ~150+ points to win).")
    print("="*50)

if __name__ == "__main__":
    # Pick a high-volume day from your dataset to test
    # Check your training_features.csv for a date with a lot of games
    run_backtest("2025-08-15")