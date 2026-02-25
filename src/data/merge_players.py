import pandas as pd
import os
import sys
from thefuzz import process, fuzz

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import src.config as config


MANUAL_CORRECTIONS = {}

def create_player_map():
    print("🔗 Starting Entity Resolution (WNBA <-> Unrivaled)...")
    print(f"   LEFT SIDE (WNBA):      {config.MERGE_WNBA_SOURCE}")
    print(f"   RIGHT SIDE (UNRIVALED): {config.MERGE_UNRIVALED_SOURCE}")

    # 1. Load Data using Config Paths
    if not os.path.exists(config.MERGE_WNBA_SOURCE):
        print(f"❌ CRITICAL ERROR: WNBA 2025 file not found at: {config.MERGE_WNBA_SOURCE}")
        print("   Did you run 'wnba_loader.py' for the 2025 season?")
        return

    df_wnba = pd.read_csv(config.MERGE_WNBA_SOURCE)
    df_unrivaled = pd.read_csv(config.MERGE_UNRIVALED_SOURCE)

    # 2. Robust Column Selection (Unchanged)
    # We look for PLAYER_NAME (API) or Player_Name (CSV)
    if 'PLAYER_NAME' in df_wnba.columns:
        wnba_lookup = df_wnba[['PLAYER_NAME', 'PLAYER_ID']].drop_duplicates()
        wnba_lookup.columns = ['Player_Name', 'Player_ID']
    elif 'Player_Name' in df_wnba.columns:
        wnba_lookup = df_wnba[['Player_Name', 'Player_ID']].drop_duplicates()
    else:
        # Fallback: Look for any column with "Name" in it
        possible = [c for c in df_wnba.columns if 'NAME' in c.upper()]
        raise KeyError(f"❌ Could not find Player Name column. Candidates: {possible}")

    unrivaled_names = df_unrivaled['player_name'].unique()
    valid_wnba_names = wnba_lookup['Player_Name'].tolist()

    print(f"   -> WNBA 2025 Roster Size: {len(valid_wnba_names)}")
    print(f"   -> Unrivaled Roster Size: {len(unrivaled_names)}")

    # 3. The Matching Engine (Unchanged logic)
    matches = []
    
    for u_name in unrivaled_names:
        
        # A. Manual Override
        if u_name in MANUAL_CORRECTIONS:
            target_name = MANUAL_CORRECTIONS[u_name]
            score = 100
            method = "Manual"
        else:
            # B. Fuzzy Match
            best_match = process.extractOne(u_name, valid_wnba_names, scorer=fuzz.token_sort_ratio)
            target_name = best_match[0]
            score = best_match[1]
            method = "Fuzzy"

        # C. Confidence Gate
        if score < 85:
            print(f"      ⚠️  Dropping '{u_name}' (Best match: '{target_name}' - Score: {score})")
            continue 

        # D. Retrieve ID
        try:
            w_id = wnba_lookup[wnba_lookup['Player_Name'] == target_name]['Player_ID'].values[0]
            matches.append({
                'unrivaled_name': u_name,
                'wnba_name': target_name,
                'wnba_id': w_id,
                'match_score': score,
                'method': method
            })
        except IndexError:
            print(f"      ❌ Error looking up ID for {target_name}")

    # 4. Save
    df_map = pd.DataFrame(matches)
    df_map.to_csv(config.PLAYER_MAP_OUTPUT, index=False)
    
    print("-" * 30)
    print(f"✅ Mapping Complete. Saved to {config.PLAYER_MAP_OUTPUT}")
    print(f"   Total Matches: {len(df_map)} / {len(unrivaled_names)}")

if __name__ == "__main__":
    create_player_map()