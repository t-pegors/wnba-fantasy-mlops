import os
import sys
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from io import StringIO

# Path magic
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config
from src.utils.data_utils import normalize_name

def get_ncaa_pedigree(player_name):
    """
    Hunts Sports-Reference CBB for the player's final college season.
    Returns estimated NCAA Fantasy Points.
    """
    # STRIP punctuation before converting spaces to dashes
    clean_url_name = player_name.lower().replace("'", "").replace(".", "").replace(" ", "-")
    url_name = f"{clean_url_name}-1"
    url = f"https://www.sports-reference.com/cbb/players/{url_name}.html"
    
    headers = {"User-Agent": "WNBA-MLOps-Pipeline/1.0"}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None # Player not found
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Grab the "Per Game" stats table
        table = soup.find('table', {'id': 'players_per_game'})
        if not table:
            return None
            
        # Use StringIO to prevent the Pandas FutureWarning
        df = pd.read_html(StringIO(str(table)))[0]
        df = df.dropna(subset=['Season'])
        final_season = df[df['Season'] != 'Career'].iloc[-1]
        
        # Calculate raw NCAA Fantasy Points (using standard DK scoring)
        pts = float(final_season.get('PTS', 0))
        reb = float(final_season.get('TRB', 0))
        ast = float(final_season.get('AST', 0))
        stl = float(final_season.get('STL', 0))
        blk = float(final_season.get('BLK', 0))
        tov = float(final_season.get('TOV', 0))
        threes = float(final_season.get('3P', 0))
        
        ncaa_fpts = (pts * 1.0) + (reb * 1.2) + (ast * 1.5) + (stl * 3.0) + (blk * 3.0) - (tov * 1.0) + (threes * 0.5)
        return ncaa_fpts

    except Exception as e:
        print(f"      ⚠️ Scrape error for {player_name}: {e}")
        return None

def build_rookie_database():
    print("🕵️‍♀️ Initiating OSINT Protocol: Hunting NCAA Pedigrees...")
    
    # Dynamically load the target list
    target_path = config.DATA_DIR / "metadata" / "target_ncaa_rookies.csv"
    
    if not target_path.exists():
        print(f"❌ Target list not found at {target_path}")
        print("Please create 'target_ncaa_rookies.csv' with a 'PLAYER_NAME' column.")
        return
        
    target_df = pd.read_csv(target_path)
    
    if 'PLAYER_NAME' not in target_df.columns:
        print("❌ Error: CSV must contain a 'PLAYER_NAME' column.")
        return
        
    target_rookies = target_df['PLAYER_NAME'].dropna().tolist()
    
    pedigree_records = []
    print(f"🎯 Loaded {len(target_rookies)} target NCAA profiles from template...")
    
    for name in target_rookies:
        clean_name = str(name).strip()
        print(f"   🔍 Scanning NCAA databases for: {clean_name.title()}...")
        ncaa_fpts = get_ncaa_pedigree(clean_name)
        
        if ncaa_fpts:
            # Apply the WNBA Translation Factor (0.65 Rookie Tax)
            wnba_proxy = round(ncaa_fpts * 0.65, 2)
            tier = 'ELITE' if wnba_proxy > 20 else 'ROTATION'
            print(f"      ✅ Found! NCAA FPTS: {ncaa_fpts:.1f} -> WNBA Proxy: {wnba_proxy} ({tier})")
            
            pedigree_records.append({
                'match_name': normalize_name(clean_name),
                'NCAA_FPTS': round(ncaa_fpts, 2),
                'WNBA_ROOKIE_PROXY': wnba_proxy,
                'PEDIGREE_TIER': tier
            })
        else:
            print(f"      ⏭️ Not found in NCAA database. Check spelling or URL format.")
            
        time.sleep(3) # DO NOT REMOVE: Be polite to Sports-Reference servers!
        
    if not pedigree_records:
        print("\n💨 Sweep complete. No records found.")
        return

    pedigree_df = pd.DataFrame(pedigree_records)
    
    # Save the final lookup table
    output_path = config.DATA_DIR / "metadata" / "rookie_proxies.csv"
    pedigree_df.to_csv(output_path, index=False)
    print(f"\n🎉 SUCCESS! Saved {len(pedigree_df)} NCAA baseline proxies to {output_path}")

if __name__ == "__main__":
    build_rookie_database()