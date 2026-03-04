import os
import sys
import time
import pandas as pd
from nba_api.stats.endpoints import commonallplayers, commonplayerinfo

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config


def build_vault_for_season(season: str) -> pd.DataFrame:
    """
    Fetches and enriches the active WNBA roster for a given season year.
    Combines the former scrape_2025_players + hydrate_2025_players into one step.

    Returns a DataFrame saved to data/metadata/player_vault_{season}.csv
    """
    print(f"\n{'='*55}")
    print(f"  Building Player Vault: Season {season}")
    print(f"{'='*55}")

    headers = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://stats.nba.com/',
    }

    # --- Step 1: Fetch full player list for the season ---
    print(f"📚 Fetching player list from nba_api (season={season})...")
    try:
        cap = commonallplayers.CommonAllPlayers(
            league_id=config.WNBA_LEAGUE_ID,
            season=season,
            is_only_current_season=0,
            headers=headers
        )
        roster_df = cap.get_data_frames()[0]
    except Exception as e:
        print(f"❌ API error fetching player list: {e}")
        return pd.DataFrame()

    if roster_df.empty:
        print(f"⚠️  No players returned for season {season}.")
        return pd.DataFrame()

    roster_df = roster_df.rename(columns={'DISPLAY_FIRST_LAST': 'PLAYER_NAME'})
    print(f"   -> {len(roster_df)} total players in WNBA database.")

    # --- Step 2: Filter to players active in this season ---
    # Players with a TEAM_ABBREVIATION were rostered during this season
    active_df = roster_df[
        roster_df['TEAM_ABBREVIATION'].notna() &
        (roster_df['TEAM_ABBREVIATION'].str.strip() != '')
    ].copy()
    print(f"   -> {len(active_df)} active players for {season} (have a team assignment).")

    # --- Step 3: Hydrate each active player with detailed metadata ---
    output_path = config.DATA_DIR / "metadata" / f"player_vault_{season}.csv"

    # Season-level skip: if vault exists and all active players are already hydrated, done.
    existing_enriched = {}
    if output_path.exists() and not config.OVERWRITE:
        existing = pd.read_csv(output_path)
        if 'POSITION' in existing.columns:
            hydrated = existing[existing['POSITION'].notna()]
            existing_enriched = hydrated.set_index('PERSON_ID').to_dict('index')
            if set(active_df['PERSON_ID']).issubset(existing_enriched.keys()):
                print(f"✅ Season {season} vault is complete. Skipping (set OVERWRITE=True to force refresh).")
                return pd.read_csv(output_path)
            print(f"   -> Resuming: {len(existing_enriched)} players already hydrated, skipping them.")

    print(f"💧 Hydrating {len(active_df)} players with position/physical/experience data...")
    enriched_rows = []

    for idx, (_, row) in enumerate(active_df.iterrows()):
        pid = row['PERSON_ID']
        name = row['PLAYER_NAME']

        # Re-use cached data if available from a prior partial run
        if pid in existing_enriched:
            print(f"   [{idx + 1}/{len(active_df)}] {name}... (skipped, already done)")
            enriched_rows.append({'PERSON_ID': pid, **existing_enriched[pid]})
            continue

        print(f"   [{idx + 1}/{len(active_df)}] {name}...")

        # Exponential backoff: up to config.MAX_RETRIES attempts
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                info = commonplayerinfo.CommonPlayerInfo(
                    player_id=pid,
                    league_id_nullable=config.WNBA_LEAGUE_ID
                )
                detail_df = info.get_data_frames()[0]

                enriched_rows.append({
                    'PERSON_ID': pid,
                    'POSITION': detail_df['POSITION'].iloc[0],
                    'HEIGHT': detail_df['HEIGHT'].iloc[0],
                    'WEIGHT': detail_df['WEIGHT'].iloc[0],
                    'SEASON_EXP': detail_df['SEASON_EXP'].iloc[0],
                    'BIRTHDATE': detail_df['BIRTHDATE'].iloc[0],
                    'COUNTRY': detail_df['COUNTRY'].iloc[0],
                    'SCHOOL': detail_df['SCHOOL'].iloc[0],
                })
                break  # Success — exit retry loop
            except Exception as e:
                wait = config.RETRY_DELAY * (2 ** (attempt - 1))  # 5s, 10s, 20s
                if attempt < config.MAX_RETRIES:
                    print(f"   ⚠️  Attempt {attempt} failed for {name}: {e}")
                    print(f"       Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"   ❌ All {config.MAX_RETRIES} attempts failed for {name}. Skipping.")

        time.sleep(0.8)  # Base rate limit between players

    # --- Step 4: Merge enrichment back onto the active roster ---
    enriched_df = pd.DataFrame(enriched_rows)
    final_df = active_df.merge(enriched_df, on='PERSON_ID', how='left')

    # --- Step 5: Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"\n✅ Vault saved: {output_path}")
    print(f"   Rows: {len(final_df)} | Columns: {len(final_df.columns)}")
    return final_df


def build_all_vaults():
    """Builds a player vault CSV for every season in config.SEASONS_TO_FETCH."""
    print(f"Building player vaults for seasons: {config.SEASONS_TO_FETCH}")
    for season in config.SEASONS_TO_FETCH:
        build_vault_for_season(season)
    print("\nAll vaults complete.")


if __name__ == "__main__":
    build_all_vaults()
