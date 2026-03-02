import requests
import pandas as pd
import os
import sys
import re

# Path magic for 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config

def scrape_espn_vault():
    print("🚀 Initiating Deep ESPN API Vault Scrape (v2.1)...")
    
    # The 'enable=roster' flag is the secret sauce here
    base_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams?enable=roster"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # ESPN's JSON structure can be deeply nested
        teams_list = data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', [])
        
        all_players = []
        
        for entry in teams_list:
            team_info = entry.get('team', {})
            team_name = team_info.get('displayName')
            # Extract athletes directly from the 'roster' key enabled by our query param
            roster = team_info.get('athlete', team_info.get('athletes', []))
            
            # If the roster is a dict (sometimes happens in v2), get the list
            if isinstance(roster, dict):
                roster = roster.get('items', [])

            print(f"🏀 Processing {len(roster)} athletes for {team_name}...")
            
            for player in roster:
                # Height parsing (e.g., "6' 4\"")
                height_str = player.get('displayHeight', '0\' 0"')
                total_inches = 0
                try:
                    parts = re.findall(r'\d+', height_str)
                    if len(parts) >= 2:
                        total_inches = (int(parts[0]) * 12) + int(parts[1])
                except:
                    total_inches = None

                all_players.append({
                    'PLAYER_NAME': player.get('fullName'),
                    'ESPN_ID': player.get('id'),
                    'POSITION': player.get('position', {}).get('abbreviation', 'U'),
                    'TEAM_NAME': team_name,
                    'HEIGHT_INCHES': total_inches,
                    'WEIGHT_LBS': player.get('weight'),
                    'EXP_YEARS': player.get('experience', {}).get('years', 0),
                    'COLLEGE': player.get('birthPlace', {}).get('city', 'Unknown'),
                    'JERSEY': player.get('jersey'),
                    'SCRAPED_AT': pd.Timestamp.now()
                })

        df = pd.DataFrame(all_players)
        
        if not df.empty:
            output_path = config.DATA_DIR / "metadata" / "player_vault.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"✅ SUCCESS! Archived {len(df)} players to {output_path}")
        else:
            print("❌ Still 0 players. The API might be in 'Offseason' mode.")
            
        return df

    except Exception as e:
        print(f"❌ Scrape failed: {e}")

if __name__ == "__main__":
    scrape_espn_vault()