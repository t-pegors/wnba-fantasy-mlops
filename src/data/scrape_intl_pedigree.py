import os
import sys
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
from pathlib import Path

# Path magic
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config
from src.utils.data_utils import normalize_name

def get_intl_pedigree(player_name):
    """
    Hunts Basketball-Reference International for the player's most recent overseas season.
    """
    clean_url_name = player_name.lower().replace("'", "").replace(".", "").replace(" ", "-")
    
    # International URL structure is slightly different
    url = f"https://www.basketball-reference.com/international/players/{clean_url_name}-1.html"
    
    headers = {"User-Agent": "WNBA-MLOps-Pipeline/1.0"}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None 
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # International tables often use different IDs, 'per_game-intl' is standard
        table = soup.find('table', {'id': lambda L: L and L.endswith('per_game-intl')})
        if not table:
            # Fallback to standard per_game if intl suffix is missing
            table = soup.find('table', {'id': 'per_game'})
            if not table:
                return None
            
        df = pd.read_html(StringIO(str(table)))[0]
        df = df.dropna(subset=['Season'])
        final_season = df[df['Season'] != 'Career'].iloc[-1]
        
        # Calculate raw Fantasy Points
        pts = float(final_season.get('PTS', 0))
        reb = float(final_season.get('TRB', 0))
        ast = float(final_season.get('AST', 0))
        stl = float(final_season.get('STL', 0))
        blk = float(final_season.get('BLK', 0))
        tov = float(final_season.get('TOV', 0))
        threes = float(final_season.get('3P', 0))
        
        intl_fpts = (pts * 1.0) + (reb * 1.2) + (ast * 1.5) + (stl * 3.0) + (blk * 3.0) - (tov * 1.0) + (threes * 0.5)
        return intl_fpts

    except Exception as e:
        print(f"      ⚠️ Scrape error for {player_name}: {e}")
        return None

def build_intl_database():
    print("🌍 Initiating OSINT Protocol: Hunting International Pedigrees...")
    
    target_path = config.DATA_DIR / "metadata" / "target_intl_rookies.csv"
    
    if not target_path.exists():
        print(f"❌ Target list not found at {target_path}")
        return
        
    target_df = pd.read_csv(target_path)
    target_rookies = target_df['PLAYER_NAME'].dropna().tolist()
    
    pedigree_records = []
    print(f"🎯 Loaded {len(target_rookies)} target International profiles...")
    
    for name in target_rookies:
        clean_name = str(name).strip()
        print(f"   🔍 Scanning International databases for: {clean_name.title()}...")
        intl_fpts = get_intl_pedigree(clean_name)
        
        if intl_fpts:
            # The International Translation Factor (0.85 Pro Tax)
            wnba_proxy = round(intl_fpts * 0.85, 2)
            tier = 'ELITE' if wnba_proxy > 20 else 'ROTATION'
            print(f"      ✅ Found! INTL FPTS: {intl_fpts:.1f} -> WNBA Proxy: {wnba_proxy} ({tier})")
            
            pedigree_records.append({
                'match_name': normalize_name(clean_name),
                'OVERSEAS_FPTS': round(intl_fpts, 2),
                'WNBA_ROOKIE_PROXY': wnba_proxy,
                'PEDIGREE_TIER': tier
            })
        else:
            print(f"      ⏭️ Not found. May require manual lookup on RealGM or FIBA.")
            
        time.sleep(3) 
        
    if not pedigree_records:
        print("\n💨 Sweep complete. No records found.")
        return

    pedigree_df = pd.DataFrame(pedigree_records)
    
    output_path = config.DATA_DIR / "metadata" / "intl_proxies.csv"
    pedigree_df.to_csv(output_path, index=False)
    print(f"\n🎉 SUCCESS! Saved {len(pedigree_df)} International baseline proxies to {output_path}")

if __name__ == "__main__":
    build_intl_database()