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
    print("🚀 Starting WNBA Feature Engineering Pipeline (Baseline Model)...")

    # 1. Load ALL Available Historical Data
    # glob finds all files matching the pattern in the directory
    search_pattern = str(config.RAW_DATA_DIR / "wnba_*_gamelogs.csv")
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        raise FileNotFoundError(f"❌ No WNBA gamelog CSVs found matching: {search_pattern}")
        
    print(f"📂 Found {len(all_files)} seasons of historical data. Merging...")
    df_list = [pd.read_csv(file) for file in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # 2. Apply Dynamic Scoring Rules
    scoring_weights = config.load_scoring_system(config.DEFAULT_SCORING_SYSTEM)
    df['FANTASY_PTS'] = df.apply(lambda row: calc_fp(row, scoring_weights), axis=1)

    # 3. Filter the Noise (Min Games Threshold)
    game_counts = df['PLAYER_ID'].value_counts()
    valid_players = game_counts[game_counts >= config.MIN_GAMES_THRESHOLD].index
    df = df[df['PLAYER_ID'].isin(valid_players)].copy()
    print(f"🧹 Filtered out players with < {config.MIN_GAMES_THRESHOLD} career games in this dataset.")

    # 4. Sort chronologically to prevent data leakage (Time Travel)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'])

    # 5. Calculate Lag Features (Rolling Averages)
    print(f"📈 Calculating {config.ROLLING_WINDOW_SHORT}-game and {config.ROLLING_WINDOW_LONG}-game rolling averages...")
    
    # We MUST use .shift(1) so today's prediction only uses data from *previous* games
    df[f'FPTS_{config.ROLLING_WINDOW_SHORT}G_AVG'] = df.groupby('PLAYER_ID')['FANTASY_PTS'].transform(
        lambda x: x.rolling(window=config.ROLLING_WINDOW_SHORT, min_periods=1).mean().shift(1)
    )
    
    df[f'FPTS_{config.ROLLING_WINDOW_LONG}G_AVG'] = df.groupby('PLAYER_ID')['FANTASY_PTS'].transform(
        lambda x: x.rolling(window=config.ROLLING_WINDOW_LONG, min_periods=1).mean().shift(1)
    )
    
    # Contextual Features: Rest Days
    df['DAYS_REST'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days.fillna(7)

    # 6. Drop rows that have NaN in our features (like the very first game of the season)
    df = df.dropna(subset=[f'FPTS_{config.ROLLING_WINDOW_SHORT}G_AVG'])

    # 7. Save the "Golden Table"
    output_path = config.PROCESSED_DATA_DIR / "training_features.csv"
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Feature Engineering Complete! Baseline dataset saved to: {output_path}")
    print(f"📊 Final Dataset Shape: {df.shape}")

if __name__ == "__main__":
    engineer_features()