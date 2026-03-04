import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path


# Path magic to import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config
from src.utils.data_utils import load_scoring_system
from src.utils.data_utils import normalize_name

#############################################################################################

def calc_fp(row, w):
    """
    Calculates Fantasy Points dynamically based on loaded weights.
    To change scoring system, add system to config/scoring/ and update in config.py
    """
    return (row['PTS'] * w['PTS']) + \
           (row['REB'] * w['REB']) + \
           (row['AST'] * w['AST']) + \
           (row['STL'] * w['STL']) + \
           (row['BLK'] * w['BLK']) + \
           (row['TOV'] * w['TOV']) + \
           (row.get('FG3M', 0) * w.get('FG3M', 0))

def engineer_features():
    print("🚀 Starting WNBA Feature Engineering Pipeline...")

    #############################################
    ## Load and Merge All Historical WNBA Data ##
    #############################################
    
    search_pattern = str(config.RAW_DATA_DIR / "wnba_*_gamelogs.csv")
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        raise FileNotFoundError(f"❌ No WNBA gamelog CSVs found matching: {search_pattern}")
        
    print(f"📂 Found {len(all_files)} seasons of historical data. Merging...")
    df_list = [pd.read_csv(file) for file in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # Apply Scoring (Target)
    scoring_weights = load_scoring_system(config.DEFAULT_SCORING_SYSTEM)
    df['FANTASY_PTS'] = df.apply(lambda row: calc_fp(row, scoring_weights), axis=1)

    # Filter players that don't exceed the minimum game threshold
    game_counts = df['PLAYER_ID'].value_counts()
    valid_players = game_counts[game_counts >= config.MIN_GAMES_THRESHOLD].index
    df = df[df['PLAYER_ID'].isin(valid_players)].copy()

    # Sort chronologically to prevent data leakage
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'])

    #####################################################
    ### Features: IS_HOME, DAYS_REST, IS_BACK_TO_BACK ###
    #####################################################

    print("🧠 Engineering predictive features...")

    # Venue Features (Home vs Away)
    # If the matchup contains ' vs. ', they are the home team. If '@', away.
    df['IS_HOME'] = np.where(df['MATCHUP'].str.contains(' vs. '), 1, 0)

    # Rest & Fatigue Features
    df['DAYS_REST'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days.fillna(7)
    df['IS_BACK_TO_BACK'] = np.where(df['DAYS_REST'] <= 1, 1, 0)

    ################################################################
    ### Features: FPTS_[short]_AVG, FPTS_[long]_AVG, FPTS_SEASON_AVG
    ################################################################

    # Rolling Average (Short)
    df[f'FPTS_{config.ROLLING_WINDOW_SHORT}G_AVG'] = df.groupby('PLAYER_ID')['FANTASY_PTS'].transform(
        lambda x: x.rolling(window=config.ROLLING_WINDOW_SHORT, min_periods=1).mean().shift(1)
    )
    
    # Rolling Average (Long)
    df[f'FPTS_{config.ROLLING_WINDOW_LONG}G_AVG'] = df.groupby('PLAYER_ID')['FANTASY_PTS'].transform(
        lambda x: x.rolling(window=config.ROLLING_WINDOW_LONG, min_periods=1).mean().shift(1)
    )

    print("🌉 Building cross-season Bayesian anchors...")
    
    # Extract the Season Year before grouping
    df['SEASON'] = df['GAME_DATE'].dt.year

    # Chronological Game Index (1 to ~40 for each season)
    df['SEASON_GAME_NUM'] = df.groupby(['PLAYER_ID', 'SEASON']).cumcount() + 1

    # Extract Prior Season Average
    # Calculate the final average for each player in each season, then shift it forward a year
    season_summaries = df.groupby(['PLAYER_ID', 'SEASON'])['FANTASY_PTS'].mean().reset_index()
    # Explicitly sort chronologically to ensure .shift(1) pulls the correct previous year
    season_summaries = season_summaries.sort_values(by=['PLAYER_ID', 'SEASON'])
    season_summaries['PRIOR_SEASON_AVG'] = season_summaries.groupby('PLAYER_ID')['FANTASY_PTS'].shift(1)
    
    # Merge the prior year's average back into the main timeline
    df = df.merge(season_summaries[['PLAYER_ID', 'SEASON', 'PRIOR_SEASON_AVG']], 
                  on=['PLAYER_ID', 'SEASON'], 
                  how='left')

    # ==============================================================================
    # 🧬 COLD START MITIGATION: OSINT ROOKIE & INTERNATIONAL PROXIES
    # ==============================================================================
    # PURPOSE:
    # Traditional rolling averages fail for players with zero historical WNBA data 
    # (e.g., incoming NCAA draft picks or international professionals). To prevent
    # the DFS optimizer from blindingly ignoring highly efficient, underpriced rookies, 
    # we inject translation-adjusted proxy baselines gathered via Open Source Intelligence.
    #
    # DATA LINEAGE & TRANSLATION MATHEMATICS:
    # - NCAA Proxies (rookie_proxies.csv): Scraped from Sports-Reference. College 
    #   production is penalized with a 0.65 multiplier to simulate the WNBA transition.
    # - INTL Proxies (intl_proxies.csv): Manual scout overrides for overseas leagues. 
    #   Professional European stats are penalized with a lighter 0.85 multiplier.
    #
    # THE FALLBACK CASCADE (DEFENSIVE PROGRAMMING):
    # To ensure the pipeline never crashes on a missing value, 'PRIOR_SEASON_AVG' 
    # is populated using a strict hierarchy of trust:
    #   1. Actual WNBA Prior Season Average (Established Veterans)
    #   2. OSINT WNBA Rookie Proxy (Tracked Draft Picks & International Pros)
    #   3. Flat 12.0 FPTS Baseline (Absolute Unknowns / Replacement Level)
    #
    # An active audit masks out historical ghosts and strictly alerts the terminal 
    # if a modern player falls through to the Tier 3 baseline.
    # ==============================================================================
    
    print("🧬 Injecting Rookie Pedigree Proxies...")
    
    # 1. Load the unified proxy database
    proxy_path = config.PROCESSED_DATA_DIR / "rookie_proxies.csv"

    proxy_dfs = []
    if proxy_path.exists():
        proxy_dfs.append(pd.read_csv(proxy_path)[['match_name', 'WNBA_ROOKIE_PROXY']])
        
    if proxy_dfs:
        # Combine NCAA and INTL proxies
        all_proxies = pd.concat(proxy_dfs, ignore_index=True)
        
        # Merge them into the main timeline
        df['match_name'] = df['PLAYER_NAME'].apply(normalize_name)
        df = df.merge(all_proxies, on='match_name', how='left')
        
        # --- 🚨 MISSING DATA AUDIT 🚨 ---
        # Find players who have NO historical data AND NO proxy data
        max_season = df['SEASON'].max()
        current_season_mask = df['SEASON'] == max_season
        unmapped_mask = df['PRIOR_SEASON_AVG'].isna() & df['WNBA_ROOKIE_PROXY'].isna() & current_season_mask
        
        unmapped_players = df[unmapped_mask]['PLAYER_NAME'].unique()
        
        if len(unmapped_players) > 0:
            print(f"   ⚠️ WARNING: {len(unmapped_players)} players have NO historical or proxy data!")
            print("   They are defaulting to a flat 12.0. Consider adding them to your OSINT targets:")
            
            # Print up to 10 names so we don't spam the terminal if there are dozens
            for p in unmapped_players[:10]:
                print(f"      - {p}")
            if len(unmapped_players) > 10:
                print(f"      ...and {len(unmapped_players) - 10} more.")
        # ------------------------------------
        
        # Cascade Fill: Try Historical -> Try OSINT Proxy -> Fallback to 12.0
        df['PRIOR_SEASON_AVG'] = df['PRIOR_SEASON_AVG'].fillna(df['WNBA_ROOKIE_PROXY']).fillna(12.0)
        
        # Clean up temporary columns
        df = df.drop(columns=['match_name', 'WNBA_ROOKIE_PROXY'])
    else:
        # Fallback if the CSVs are completely missing
        print("   ⚠️ WARNING: No proxy CSVs found in data/curated/. All missing players defaulting to 12.0.")
        df['PRIOR_SEASON_AVG'] = df['PRIOR_SEASON_AVG'].fillna(12.0)

    # Season-to-Date Average (Bayesian Blend for Early Season)
    # - For the first set of games (before all rolling windows can be calculated), data is 
    #   wrapped in from the previous season
    # - Any missing rolling window averages are replaced with this blended average
    # Get the raw expanding mean of the CURRENT season (excluding today)
    df['CURRENT_SEASON_AVG'] = df.groupby(['PLAYER_ID', 'SEASON'])['FANTASY_PTS'].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    # Calculate the "Trust Weight" (0.0 on Game 1, 1.0 by Game 11)
    # ROLLING_WINDOW_LONG is the "burn-in" period.
    df['NEW_SEASON_WEIGHT'] = ((df['SEASON_GAME_NUM'] - 1) / config.ROLLING_WINDOW_LONG).clip(upper=1.0)

    # Fill Game 1's NaN with 0 temporarily so the math doesn't break (weight is 0 anyway)
    df['CURRENT_SEASON_AVG'] = df['CURRENT_SEASON_AVG'].fillna(0)

    # The Bayesian Blend: (Current * Weight) + (Prior * Inverse Weight)
    df['FPTS_SEASON_AVG'] = (df['CURRENT_SEASON_AVG'] * df['NEW_SEASON_WEIGHT']) + \
                            (df['PRIOR_SEASON_AVG'] * (1 - df['NEW_SEASON_WEIGHT']))
    
    # Replace missing short-term/long-term rolling averages with Blended Average (i.e. don't drop early games)
    short_col = f'FPTS_{config.ROLLING_WINDOW_SHORT}G_AVG'
    long_col = f'FPTS_{config.ROLLING_WINDOW_LONG}G_AVG'

    df[short_col] = df[short_col].fillna(df['FPTS_SEASON_AVG'])
    df[long_col] = df[long_col].fillna(df['FPTS_SEASON_AVG'])

    #########################################################
    ### Features: TEAM_WIN_PCT, OPP_WIN_PCT, WIN_PCT_DIFF ###
    #########################################################

    print("📈 Calculating chronological team standings and opponent strength...")

    # Extract Opponent Abbreviation from MATCHUP (e.g., "PHO vs. LVA" -> "LVA")
    df['OPP_ABBREVIATION'] = df['MATCHUP'].str.split(' ').str[-1]

    # Build a clean, chronological table of every team's games
    team_games = df[['TEAM_ABBREVIATION', 'GAME_DATE', 'WL', 'SEASON']].drop_duplicates().copy()
    team_games = team_games.sort_values(by=['TEAM_ABBREVIATION', 'GAME_DATE'])

    # Double-check the date format on the new dataframe
    team_games['GAME_DATE'] = pd.to_datetime(team_games['GAME_DATE'])

    # Convert 'W'/'L' to 1/0
    team_games['WIN_FLAG'] = np.where(team_games['WL'] == 'W', 1, 0)
    
    # Calculate current season rolling win percentage
    team_games['CURRENT_WIN_PCT'] = team_games.groupby(['TEAM_ABBREVIATION', 'SEASON'])['WIN_FLAG'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # Extract Prior Season Final Win Percentage
    team_summaries = team_games.groupby(['TEAM_ABBREVIATION', 'SEASON'])['WIN_FLAG'].mean().reset_index()
    team_summaries = team_summaries.sort_values(by=['TEAM_ABBREVIATION', 'SEASON'])
    team_summaries['PRIOR_WIN_PCT'] = team_summaries.groupby('TEAM_ABBREVIATION')['WIN_FLAG'].shift(1).fillna(0.500) # 0.500 for new expansion teams
    
    # Merge the prior win percentage back
    team_games = team_games.merge(team_summaries[['TEAM_ABBREVIATION', 'SEASON', 'PRIOR_WIN_PCT']], 
                                  on=['TEAM_ABBREVIATION', 'SEASON'], how='left')
    
    # Calculate Team Game Number to apply Bayesian Weight
    team_games['TEAM_GAME_NUM'] = team_games.groupby(['TEAM_ABBREVIATION', 'SEASON']).cumcount() + 1
    team_games['TEAM_WEIGHT'] = ((team_games['TEAM_GAME_NUM'] - 1) / config.ROLLING_WINDOW_LONG).clip(upper=1.0)
    
    team_games['CURRENT_WIN_PCT'] = team_games['CURRENT_WIN_PCT'].fillna(0)
    
    # The Team Bayesian Blend
    team_games['TEAM_WIN_PCT'] = (team_games['CURRENT_WIN_PCT'] * team_games['TEAM_WEIGHT']) + \
                                 (team_games['PRIOR_WIN_PCT'] * (1 - team_games['TEAM_WEIGHT']))
    
    # Merge the Player's Team Win PCT back into the main dataframe
    df = df.merge(team_games[['TEAM_ABBREVIATION', 'GAME_DATE', 'TEAM_WIN_PCT']], 
                  on=['TEAM_ABBREVIATION', 'GAME_DATE'], 
                  how='left')
                  
    # Merge the Opponent's Team Win PCT into the main dataframe
    opp_games = team_games[['TEAM_ABBREVIATION', 'GAME_DATE', 'TEAM_WIN_PCT']].rename(
        columns={'TEAM_ABBREVIATION': 'OPP_ABBREVIATION', 'TEAM_WIN_PCT': 'OPP_WIN_PCT'}
    )
    df = df.merge(opp_games, on=['OPP_ABBREVIATION', 'GAME_DATE'], how='left')

    # Calculate the Matchup Differential 
    # Positive = Our team is better. Negative = Opponent is better. Zero = Evenly matched.
    df['WIN_PCT_DIFF'] = df['TEAM_WIN_PCT'] - df['OPP_WIN_PCT']
    
    

    ################### 
    # Cleanup and Save
    ################### 

    print("🧹 Cleaning up raw stats and non-predictive columns...")
    
    # We drop the raw stats because they happen DURING the game. 
    # If the model sees them, it's cheating (Data Leakage).
    leaky_box_score_stats = [
        'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'FG_PCT', 
        'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'PF', 
        'PLUS_MINUS', 'MIN', 'MATCHUP', 'WL' 
    ]
    
    # Safely drop ONLY the leaky stats. 
    # We explicitly do NOT drop config.META_COLUMNS here so they survive into the CSV.
    actual_drops = [col for col in leaky_box_score_stats if col in df.columns]
    df = df.drop(columns=actual_drops)

    # Save the "Golden Table"
    output_path = config.PROCESSED_DATA_DIR / "training_features.csv"
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Feature Engineering Complete! Baseline dataset saved to: {output_path}")
    print(f"📊 Final Dataset Shape: {df.shape}")
    print(f"📝 Columns preserved: {list(df.columns)}")

if __name__ == "__main__":
    engineer_features()