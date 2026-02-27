import os
from pathlib import Path
import yaml

# --- PROJECT PATHS ---
# Automatically locate the root of the project relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# --- DATA INGESTION SETTINGS ---
# WNBA League ID is always '10' in nba_api
WNBA_LEAGUE_ID = '10'

# List of seasons to fetch. 
# WNBA seasons are typically referenced by year (e.g., '2024').
# You can add past seasons here to build a historical dataset.
SEASONS_TO_FETCH = ['2021', '2022'] #, '2023', '2024', '2025']

# --- PIPELINE CONTROL ---
# Set to True if you want to force a re-download of existing data
OVERWRITE = True

# API Retry Settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds

# --- ENTITY RESOLUTION CONFIG ---
# The specific files we compare to create the Master Player Map
# We use 2025 because it contains the most recent active roster including 2025 rookies
MERGE_WNBA_SOURCE = RAW_DATA_DIR / "wnba_2025_gamelogs.csv"
MERGE_UNRIVALED_SOURCE = PROCESSED_DATA_DIR / "unrivaled_2025_processed.csv"
PLAYER_MAP_OUTPUT = PROCESSED_DATA_DIR / "player_mapping.csv"

# --- MODELING & FEATURE ENGINEERING PARAMETERS ---
# Minimum number of games a player must play to be included
MIN_GAMES_THRESHOLD = 10 

# Feature Engineering: Rolling Average Windows
ROLLING_WINDOW_SHORT = 3
ROLLING_WINDOW_LONG = 10

# Default rulebook to use if none is specified
DEFAULT_SCORING_SYSTEM = 'wnba_default'

# --- SCORING CONFIGURATION --

SCORING_DIR = Path(__file__).resolve().parent.parent / "config" / "scoring"

def load_scoring_system(system_name=DEFAULT_SCORING_SYSTEM):
    """
    Loads a scoring configuration from the config/scoring directory.
    Usage: rules = load_scoring_system('fanduel_dfs')
    """
    filepath = SCORING_DIR / f"{system_name}.yml"
    
    if not filepath.exists():
        raise FileNotFoundError(f"❌ Scoring system '{system_name}' not found at {filepath}")
        
    with open(filepath, 'r') as file:
        config_data = yaml.safe_load(file)
        
    print(f"Loaded Scoring System: {config_data['name']}")
    return config_data['weights']
