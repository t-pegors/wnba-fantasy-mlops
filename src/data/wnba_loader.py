import pandas as pd
import os
import time
from datetime import datetime
from nba_api.stats.endpoints import leaguegamelog
from requests.exceptions import ReadTimeout, ConnectionError

# Import our new central config
import sys
import os

# Add the project root to python path so we can import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import src.config as config

def fetch_season_data(season):
    """
    Fetches data for a single season with retry logic.
    Returns a DataFrame or None if failed.
    """
    print(f"   -> Fetching {season} season data...")
    
    for attempt in range(config.MAX_RETRIES):
        try:
            # 1. Fetch Data using nba_api
            log = leaguegamelog.LeagueGameLog(
                league_id=config.WNBA_LEAGUE_ID, 
                season=season,
                player_or_team_abbreviation='P' 
            )
            
            df = log.get_data_frames()[0]
            
            if df.empty:
                print(f"      ⚠️  Warning: No data found for {season}.")
                return None

            # 2. Add Metadata
            df['scraped_at'] = datetime.now().isoformat
            df['season_id'] = season
            
            return df

        except (ReadTimeout, ConnectionError) as e:
            print(f"      ❌ Network Error: {e}")
            if attempt < config.MAX_RETRIES - 1:
                print(f"      ...Waiting {config.RETRY_DELAY}s...")
                time.sleep(config.RETRY_DELAY)
            else:
                print("      ❌ Max retries reached.")
                return None
        except Exception as e:
            print(f"      ❌ Unexpected Error: {e}")
            return None

def main():
    """
    Main orchestration function.
    Checks for existing files and loops through configured seasons.
    """
    print(f"🏀 Starting WNBA Data Pipeline")
    print(f"   Target Seasons: {config.SEASONS_TO_FETCH}")
    print(f"   Overwrite Mode: {config.OVERWRITE}")
    
    # Ensure raw directory exists
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

    for season in config.SEASONS_TO_FETCH:
        # Define the output path for this specific season
        output_filename = f"wnba_{season}_gamelogs.csv"
        output_path = os.path.join(config.RAW_DATA_DIR, output_filename)

        # CHECK: Does file exist?
        if os.path.exists(output_path) and not config.OVERWRITE:
            print(f"⏭️  Skipping {season}: File exists at {output_filename}")
            continue
        
        # FETCH: Download the data
        df = fetch_season_data(season)
        
        # SAVE: Write to CSV
        if df is not None:
            df.to_csv(output_path, index=False)
            print(f"✅ Saved {season} data to {output_filename} ({len(df)} rows)")

if __name__ == "__main__":
    main()