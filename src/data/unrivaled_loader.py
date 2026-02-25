import pandas as pd
import requests
import os
import sys
from datetime import datetime
from io import StringIO

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import src.config as config

# NEW TARGET URL: The full player list, not the dashboard
URL = "https://www.unrivaled.basketball/stats/player"

def fetch_unrivaled_stats():
    print(f"🕵️‍♀️ Infiltrating Unrivaled at: {URL}...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()
        
        # Parse all tables
        distinct_tables = pd.read_html(StringIO(response.text))
        print(f"✅ Extraction Complete. Found {len(distinct_tables)} tables.")

        stats_df = None
        
        # SMARTER SEARCH: Look for a table with "GP" (Games Played) and "PTS"
        # This filters out the small leaderboard widgets
        for i, df in enumerate(distinct_tables):
            # Normalize column names to uppercase for matching
            cols = [str(c).upper().strip() for c in df.columns]
            print(f"   -> Table {i} Columns: {cols[:5]}...")
            
            if 'PLAYER' in cols and 'GP' in cols:
                stats_df = df
                print(f"   🎯 TARGET ACQUIRED: Table {i} is the Master Stats List.")
                break
        
        if stats_df is None:
            print("❌ Failure: Could not find the master table. Checking for backups...")
            # Fallback: Sometimes headers are MultiIndex. Let's try to flatten them.
            if len(distinct_tables) > 0:
                print("   ⚠️  Attempting to use the largest table found...")
                stats_df = max(distinct_tables, key=len) # Pick the biggest one
        
        if stats_df is None:
            return

        # Data Cleaning
        # 1. Ensure we treat the header correctly (sometimes pandas messes up 2-row headers)
        if isinstance(stats_df.columns, pd.MultiIndex):
            stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]

        # 2. Add Metadata
        stats_df['scraped_at'] = datetime.now().isoformat()
        stats_df['season_id'] = '2025'
        
        # 3. Save
        os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
        output_path = os.path.join(config.RAW_DATA_DIR, "unrivaled_2025_stats.csv")
        stats_df.to_csv(output_path, index=False)
        
        print(f"💾 Mission Accomplished. Data saved to:")
        print(f"   {output_path}")
        print(f"   Rows: {len(stats_df)}")
        print(f"   Sample: {list(stats_df.columns)[:5]}")

    except Exception as e:
        print(f"❌ Error extracting data: {e}")

if __name__ == "__main__":
    fetch_unrivaled_stats()