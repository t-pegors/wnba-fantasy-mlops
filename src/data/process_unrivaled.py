import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import src.config as config

def process_unrivaled():
    print("🧹 Starting Unrivaled Data Normalization...")
    
    # 1. Load Raw Data
    raw_path = os.path.join(config.RAW_DATA_DIR, "unrivaled_2025_stats.csv")
    if not os.path.exists(raw_path):
        print(f"❌ Error: File not found at {raw_path}")
        return

    df = pd.read_csv(raw_path)
    print(f"   -> Loaded {len(df)} rows.")

    # 2. Rename Columns (Map Unrivaled -> WNBA Standard)
    # Note: Adjust the 'keys' below based on exactly what your CSV has
    column_map = {
        'PLAYER': 'player_name',
        'GP': 'games_played',
        'PTS': 'PTS',
        'REB': 'REB',
        'AST': 'AST',
        'STL': 'STL',
        'BLK': 'BLK',
        'TO': 'TOV',  # WNBA uses TOV for turnovers
        '3PM': 'FG3M' # WNBA uses FG3M for 3-pointers made
    }
    
    # Rename only columns that exist
    df = df.rename(columns=column_map)
    
    # 3. Calculate "Per Game" Stats (Since Unrivaled is often Totals)
    # If stats are totals, we divide by GP. If they are already per game, skip this.
    # We'll assume they are Per Game if "PTS" < 40 on average, otherwise Totals.
    avg_pts = df['PTS'].mean()
    if avg_pts > 40: 
        print(f"   ⚠️ Detected TOTALS (Avg PTS: {avg_pts:.1f}). Converting to PER GAME...")
        stats_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M']
        for col in stats_cols:
            if col in df.columns:
                df[f'{col}_per_game'] = df[col] / df['games_played']
    else:
        print(f"   ℹ️ Detected PER GAME stats (Avg PTS: {avg_pts:.1f}). Keeping as is.")

    # 4. Standardize Names (Remove special chars, trim spaces)
    df['player_name'] = df['player_name'].str.strip()
    
    # 5. Save to Processed
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(config.PROCESSED_DATA_DIR, "unrivaled_2025_processed.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Saved normalized data to: {output_path}")

if __name__ == "__main__":
    process_unrivaled()