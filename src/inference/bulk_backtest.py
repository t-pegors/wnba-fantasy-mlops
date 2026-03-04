import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings

# Suppress pulp/xgboost warnings for a clean terminal output
warnings.filterwarnings('ignore')

# Path magic 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config
from src.inference.backtest import normalize_name, normalize_position, solve_lineup

def calculate_dynamic_salary(row, short_window_col):
    """
    Simulates DraftKings' asymmetric pricing algorithm.
    Quick to inflate prices on hot streaks; slow to discount on cold streaks.
    """
    base_fpts = row.get('FPTS_SEASON_AVG', 0)
    
    recent_fpts = row.get(short_window_col, base_fpts)
    if pd.isna(recent_fpts):
        recent_fpts = base_fpts
        
    base_salary = base_fpts * 250
    
    if base_fpts > 0:
        form_ratio = recent_fpts / base_fpts
    else:
        form_ratio = 1.0
        
    if form_ratio > 1.10: 
        inflation_factor = min(form_ratio, 1.25)
        adjusted_salary = base_salary * inflation_factor
    elif form_ratio < 0.85: 
        discount_factor = max(form_ratio, 0.90) 
        adjusted_salary = base_salary * discount_factor
    else:
        adjusted_salary = base_salary
        
    return max(3500, min(12000, adjusted_salary))

def run_bulk_backtest(target_month="2025-05"):
    print("\n" + "="*60)
    print(f"📊 INITIATING BULK BACKTEST FOR TARGET MONTH: {target_month}")
    print("="*60)
    
    # 1. Load the Pipeline
    try:
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(project_root, "src", "models", "model.ubj"))
    except:
        print("❌ Could not load XGBoost model.")
        return

    df = pd.read_csv(config.PROCESSED_DATA_DIR / "training_features.csv")

    # Pre-load all available season vaults, keyed by year string
    vault_by_year = {}
    for season in config.SEASONS_TO_FETCH:
        path = config.DATA_DIR / "metadata" / f"player_vault_{season}.csv"
        if path.exists():
            vdf = pd.read_csv(path)
            vdf['match_name'] = vdf['PLAYER_NAME'].apply(normalize_name)
            vault_by_year[season] = vdf
    if not vault_by_year:
        print("⚠️ No player vaults found. Run build_player_vault.py first!")
        return

    # 2. Filter to the Target Timeframe
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    month_df = df[df['GAME_DATE'].dt.strftime('%Y-%m') == target_month].copy()

    if month_df.empty:
        print(f"❌ No games found in {target_month}.")
        return
    
    # 3. Financial Tracking Variables
    entry_fee = 5.00
    payout_cash_line = 150.00  
    payout_reward = 10.00      
    
    total_slates = 0
    total_wins = 0
    total_spent = 0
    total_earned = 0
    
    unique_dates = sorted(month_df['GAME_DATE'].dt.strftime('%Y-%m-%d').unique())
    
    # === THE UP-FRONT REPORT ===
    print(f"📅 MONTH OVERVIEW:")
    print(f"   -> Total Slates (Game Dates) Found: {len(unique_dates)}")
    print(f"   -> Total Player Rows to Process: {len(month_df)}")
    print("-" * 60)
    
    # 4. The Simulation Loop
    for date_str in unique_dates:
        day_df = month_df[month_df['GAME_DATE'].dt.strftime('%Y-%m-%d') == date_str].copy()
        
        # Hydrate & Normalize
        year = date_str[:4]
        vault_df = vault_by_year.get(year, next(iter(vault_by_year.values())))
        day_df['match_name'] = day_df['PLAYER_NAME'].apply(normalize_name)
        day_df = day_df.merge(vault_df[['match_name', 'POSITION']], on='match_name', how='left')
        day_df['POSITION'] = day_df['POSITION'].apply(normalize_position)
        
        # === SLATE AUDIT: Pool Breakdown ===
        # Assuming build_features assigned 12.0 as the rookie baseline
        rookie_mask = day_df['PRIOR_SEASON_AVG'] == 12.0
        num_rookies = rookie_mask.sum()
        num_vets = len(day_df) - num_rookies
        
        num_guards = (day_df['POSITION'].str.contains('G')).sum()
        num_forwards = (day_df['POSITION'].str.contains('F')).sum()
        num_teams = day_df['TEAM_ABBREVIATION'].nunique()
        approx_games = num_teams // 2

        print(f"\n[SLATE: {date_str}] 🏀 ~{approx_games} Games ({num_teams} Teams)")
        print(f"   Pool Audit: {len(day_df)} Total Players | {num_vets} Veterans | {num_rookies} Rookies/Missing")
        print(f"   Pos. Audit: {num_guards} Guards | {num_forwards} Forwards")
        
        # Check DFS valid game constraints
        if approx_games < 2:
            print("   ⚠️ WARNING: Single game slate detected. Skipping optimization.")
            continue

        # Check Position constraints before sending to PuLP
        if num_guards < 2 or num_forwards < 3:
            print("   ⚠️ WARNING: Insufficient positional depth for a valid DraftKings roster. Skipping.")
            continue

        # Proxy Salaries
        short_window_col = f"FPTS_{config.ROLLING_WINDOW_SHORT}G_AVG"
        day_df['salary'] = day_df.apply(lambda row: calculate_dynamic_salary(row, short_window_col), axis=1)
        
        # Feature isolation for XGBoost
        to_drop = config.DROPPED_FEATURES + [config.TARGET_COL, 'match_name', 'POSITION', 'salary']
        actual_drops = [c for c in to_drop if c in day_df.columns]
        X_predict = day_df.drop(columns=actual_drops).select_dtypes(include=['number'])
        
        # Predict
        day_df['predicted_pts'] = model.predict(X_predict)
        
        # Optimize
        try:
            optimal_lineup = solve_lineup(day_df.reset_index(drop=True))
            actual_score = optimal_lineup[config.TARGET_COL].sum()
            salary_used = optimal_lineup['salary'].sum()
            
            # Print the drafted lineup
            print("   🏆 DRAFTED LINEUP:")
            for _, player in optimal_lineup.iterrows():
                print(f"      - {player['match_name']:<20} | {player['POSITION']:<15} | ${player['salary']:,.0f}")
            
            # Tally Financials
            total_slates += 1
            total_spent += entry_fee
            
            if actual_score >= payout_cash_line:
                total_wins += 1
                total_earned += payout_reward
                status = "✅ WIN "
            else:
                status = "❌ LOSS"
                
            print(f"   🏁 RESULT: {status} | Actual Pts: {actual_score:6.2f} | Salary Used: ${salary_used:,.0f}/$50,000")
            
        except Exception as e:
            print(f"   ⚠️ Optimization Failed: {e}")
            continue

    # 5. The Executive Report
    if total_slates > 0:
        print("\n" + "="*60)
        print(f"📈 BULK BACKTEST REPORT: {target_month}")
        print("="*60)
        print(f"Total Valid Slates Played:  {total_slates}")
        print(f"Lineup Win Rate:            {(total_wins/total_slates)*100:.1f}%")
        print(f"Total Capital Risked:       ${total_spent:.2f}")
        print(f"Total Gross Revenue:        ${total_earned:.2f}")
        print("-" * 60)
        net_profit = total_earned - total_spent
        if net_profit > 0:
            print(f"💰 NET PROFIT:               +${net_profit:.2f}")
        else:
            print(f"🔻 NET LOSS:                  ${net_profit:.2f}")
        print("="*60)

if __name__ == "__main__":
    run_bulk_backtest("2025-05")