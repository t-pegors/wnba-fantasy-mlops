"""
Diagnostic: compare players in wnba_2025_gamelogs.csv vs player_vault_2025.csv.
Explains the gap between "Players w/ Game Appearances" and "Currently Rostered (2025)".

Run: python tests/player_vault_diagnostic.py
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.utils.data_utils import normalize_name

# Load
gamelogs = pd.read_csv(config.RAW_DATA_DIR / f"wnba_{config.CURRENT_SEASON}_gamelogs.csv")
vault    = pd.read_csv(config.ROSTERS_DATA_DIR / f"player_vault_{config.CURRENT_SEASON}.csv")

# Normalize names for matching
gamelogs['norm'] = gamelogs['PLAYER_NAME'].apply(normalize_name)
vault['norm']    = vault['PLAYER_NAME'].apply(normalize_name)

gamelog_names = set(gamelogs['norm'].unique())
vault_names   = set(vault['norm'].unique())

in_logs_not_vault = gamelog_names - vault_names   # played but not in vault
in_vault_not_logs = vault_names - gamelog_names   # in vault but never played

# Summary
print(f"\n{'='*60}")
print(f"  Player List Diagnostic — {config.CURRENT_SEASON} Season")
print(f"{'='*60}")
print(f"  Game log unique players : {len(gamelog_names)}")
print(f"  Vault unique players    : {len(vault_names)}")
print(f"  Overlap (matched)       : {len(gamelog_names & vault_names)}")
print(f"  In logs, NOT in vault   : {len(in_logs_not_vault)}")
print(f"  In vault, NOT in logs   : {len(in_vault_not_logs)}")

# Detail: played but missing from vault
if in_logs_not_vault:
    print(f"\n--- Players who PLAYED but are NOT in the vault ({len(in_logs_not_vault)}) ---")
    missing = (
        gamelogs[gamelogs['norm'].isin(in_logs_not_vault)]
        .groupby('PLAYER_NAME')
        .agg(Games=('GAME_DATE', 'nunique'))
        .sort_values('Games', ascending=False)
        .reset_index()
    )
    print(missing.to_string(index=False))

# Detail: in vault but no game appearances
if in_vault_not_logs:
    print(f"\n--- Players in vault with NO game appearances ({len(in_vault_not_logs)}) ---")
    no_games = vault[vault['norm'].isin(in_vault_not_logs)][['PLAYER_NAME', 'TEAM_ABBREVIATION', 'POSITION']]
    print(no_games.to_string(index=False))
