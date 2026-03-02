import os
import sys
import pandas as pd
from nba_api.stats.endpoints import commonallplayers
from pathlib import Path

# Path magic to import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config

def get_2025_wnba_vault():
    print("📚 Fetching 2025 WNBA Roster (Attempting Latest Parameter Names)...")
    
    headers = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://stats.nba.com/',
    }

    try:
        # LeagueID '10' is WNBA
        # is_only_current_season=0 (0 = All players, 1 = Only active)
        cap = commonallplayers.CommonAllPlayers(
            league_id='10', 
            season='2025',
            is_only_current_season=0, 
            headers=headers
        )
        
        df = cap.get_data_frames()[0]
        
        if df.empty:
            print("⚠️ API returned no data for 2025. Falling back to 2024...")
            cap = commonallplayers.CommonAllPlayers(league_id='10', season='2024', is_only_current_season=0, headers=headers)
            df = cap.get_data_frames()[0]

        # Standardizing column name for your pipeline
        df = df.rename(columns={'DISPLAY_FIRST_LAST': 'PLAYER_NAME'})

        output_path = config.DATA_DIR / "metadata" / "player_vault_2025.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"✅ Success! Captured {len(df)} players in the 2025 Vault.")
        return df

    except Exception as e:
        print(f"❌ API Error: {e}")
        print("\n💡 Troubleshooting Tip:")
        print("If it says 'unexpected keyword argument', the NBA changed the name again.")
        print("Try removing 'is_only_current_season=0' entirely from the call.")

if __name__ == "__main__":
    get_2025_wnba_vault()