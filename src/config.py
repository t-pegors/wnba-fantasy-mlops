import os
from pathlib import Path

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
SEASONS_TO_FETCH = ['2023', '2024', '2025']

# --- PIPELINE CONTROL ---
# Set to True if you want to force a re-download of existing data
OVERWRITE = False

# API Retry Settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds

# --- ENTITY RESOLUTION CONFIG ---
# The specific files we compare to create the Master Player Map
# We use 2025 because it contains the most recent active roster including 2025 rookies
MERGE_WNBA_SOURCE = RAW_DATA_DIR / "wnba_2025_gamelogs.csv"
MERGE_UNRIVALED_SOURCE = PROCESSED_DATA_DIR / "unrivaled_2025_processed.csv"
PLAYER_MAP_OUTPUT = PROCESSED_DATA_DIR / "player_mapping.csv"