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

def calc_fp(row, w):
    """Calculates Fantasy Points dynamically based on loaded weights."""
    return (row['PTS'] * w['PTS']) + \
           (row['REB'] * w['REB']) + \
           (row['AST'] * w['AST']) + \
           (row['STL'] * w['STL']) + \
           (row['BLK'] * w['BLK']) + \
           (row['TOV'] * w['TOV']) + \
           (row.get('FG3M', 0) * w.get('FG3M', 0))

def engineer_features():
    print("🚀 Starting WNBA Feature Engineering Pipeline...")

    # 1. Load ALL Available Historical Data
    search_pattern = str(config.RAW_DATA_DIR / "wnba_*_gamelogs.csv")
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        raise FileNotFoundError(f"❌ No WNBA gamelog CSVs found matching: {search_pattern}")
        
    print(f"📂 Found {len(all_files)} seasons of historical data. Merging...")
    df_list = [pd.read_csv(file) for file in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # 2. Apply Dynamic Scoring Rules (The Target)
    scoring_weights = config.load_scoring_system(config.DEFAULT_SCORING_SYSTEM)
    df['FANTASY_PTS'] = df.apply(lambda row: calc_fp(row, scoring_weights), axis=1)

    # 3. Filter the Noise (Min Games Threshold)
    game_counts = df['PLAYER_ID'].value_counts()
    valid_players = game_counts[game_counts >= config.MIN_GAMES_THRESHOLD].index
    df = df[df['PLAYER_ID'].isin(valid_players)].copy()

    # 4. Sort chronologically to prevent data leakage
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'])

    # ==========================================
    # 5. FEATURE ENGINEERING BLOCK
    # ==========================================
    print("🧠 Engineering advanced predictive features...")

    # A. Venue Features (Home vs Away)
    # If the matchup contains ' vs. ', they are the home team. If '@', away.
    df['IS_HOME'] = np.where(df['MATCHUP'].str.contains(' vs. '), 1, 0)

    # B. Rest & Fatigue Features
    df['DAYS_REST'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days.fillna(7)
    df['IS_BACK_TO_BACK'] = np.where(df['DAYS_REST'] <= 1, 1, 0)

    # C. Rolling Averages (Short and Long Term Form)
    # MUST use .shift(1) so today's prediction only uses past data
    df[f'FPTS_{config.ROLLING_WINDOW_SHORT}G_AVG'] = df.groupby('PLAYER_ID')['FANTASY_PTS'].transform(
        lambda x: x.rolling(window=config.ROLLING_WINDOW_SHORT, min_periods=1).mean().shift(1)
    )
    
    df[f'FPTS_{config.ROLLING_WINDOW_LONG}G_AVG'] = df.groupby('PLAYER_ID')['FANTASY_PTS'].transform(
        lambda x: x.rolling(window=config.ROLLING_WINDOW_LONG, min_periods=1).mean().shift(1)
    )

    # D. Season-to-Date Anchor (Our best baseline!)
    df['SEASON'] = df['GAME_DATE'].dt.year
    df['FPTS_SEASON_AVG'] = df.groupby(['PLAYER_ID', 'SEASON'])['FANTASY_PTS'].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    # E. Current Team Standing (Rolling Win Percentage)
    print("📈 Calculating chronological team standings...")
    
    # 1. Isolate unique team games to avoid player-row duplication skewing the math
    team_games = df[['TEAM_ABBREVIATION', 'GAME_DATE', 'WL', 'SEASON']].drop_duplicates().sort_values(by=['TEAM_ABBREVIATION', 'GAME_DATE'])
    
    # 2. Convert 'W'/'L' text to 1/0 integers for math
    team_games['WIN_FLAG'] = np.where(team_games['WL'] == 'W', 1, 0)
    
    # 3. Calculate expanding season win percentage (MUST shift by 1 to prevent target leakage)
    team_games['TEAM_WIN_PCT'] = team_games.groupby(['TEAM_ABBREVIATION', 'SEASON'])['WIN_FLAG'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0.00) # Give them a 0.00 win pct for the very first game of the season
    
    # 4. Merge this new feature back into our main Golden Table
    df = df.merge(team_games[['TEAM_ABBREVIATION', 'GAME_DATE', 'TEAM_WIN_PCT']], 
                  on=['TEAM_ABBREVIATION', 'GAME_DATE'], 
                  how='left')
    
    # 6. Drop rows with missing lag features (e.g., first game of the season)
    df = df.dropna(subset=[f'FPTS_{config.ROLLING_WINDOW_SHORT}G_AVG', 'FPTS_SEASON_AVG'])

    # ==========================================
    # 7. THE CLEAN-UP PHASE (Dropping the Noise)
    # ==========================================
    print("🧹 Cleaning up raw stats and non-predictive columns...")
    
    # We drop the raw stats because they happen DURING the game. 
    # If the model sees them, it's cheating (Data Leakage).
    leaky_box_score_stats = [
        'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'FG_PCT', 
        'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'PF', 
        'PLUS_MINUS', 'MIN'
    ]
    
    # We drop metadata that has no mathematical value to the algorithm
    useless_metadata = ['TEAM_ABBREVIATION', 'TEAM_NAME', 'MATCHUP', 'WL', 'VIDEO_AVAILABLE', 'scraped_at', 'SCRAPED_AT']
    
    cols_to_drop = leaky_box_score_stats + useless_metadata
    
    # Safely drop only the columns that actually exist in the dataframe
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 8. Save the "Golden Table"
    output_path = config.PROCESSED_DATA_DIR / "training_features.csv"
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Feature Engineering Complete! Baseline dataset saved to: {output_path}")
    print(f"📊 Final Dataset Shape: {df.shape}")
    print(f"📝 Columns preserved: {list(df.columns)}")

if __name__ == "__main__":
    engineer_features()