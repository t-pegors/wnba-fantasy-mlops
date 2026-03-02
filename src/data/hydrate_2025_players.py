import time
import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo
import sys
import os

# Path magic to import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config


def hydrate_active_players():
    # 1. Load the big list
    vault_df = pd.read_csv(config.DATA_DIR / "metadata" / "player_vault_2025.csv")
    
    # 2. Filter for 2025/2026 active players only (reduces calls from 1000+ to ~150)
    # We look for players associated with a team in the 2025 record
    active_df = vault_df[vault_df['TEAM_ABBREVIATION'].notna()].copy()
    
    print(f"💧 Hydrating {len(active_df)} active players with Position/Height...")
    
    hydrated_data = []
    for i, row in active_df.iterrows():
        pid = row['PERSON_ID']
        print(f"📡 Fetching details for {row['PLAYER_NAME']}...")
        
        try:
            # Individual Detail Call
            info = commonplayerinfo.CommonPlayerInfo(player_id=pid, league_id_nullable='10')
            detail_df = info.get_data_frames()[0]
            
            # Extract the 'Gold' metadata
            hydrated_data.append({
                'PERSON_ID': pid,
                'POSITION': detail_df['POSITION'].iloc[0],
                'HEIGHT': detail_df['HEIGHT'].iloc[0],
                'WEIGHT': detail_df['WEIGHT'].iloc[0],
                'SEASON_EXP': detail_df['SEASON_EXP'].iloc[0]
            })
            
            # CRITICAL: Sleep to avoid the NBA Firewall "Ban Hammer"
            time.sleep(0.8) 
            
        except Exception as e:
            print(f"⚠️ Failed {row['PLAYER_NAME']}: {e}")
            continue

    # 3. Merge and Save
    details_df = pd.DataFrame(hydrated_data)
    final_vault = vault_df.merge(details_df, on='PERSON_ID', how='left', suffixes=('', '_detailed'))
    
    final_vault.to_csv(config.DATA_DIR / "metadata" / "player_vault_final.csv", index=False)
    print("✅ Final Hydrated Vault saved!")

if __name__ == "__main__":
    hydrate_active_players()