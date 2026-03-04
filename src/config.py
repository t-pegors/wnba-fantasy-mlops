import os
from pathlib import Path
import yaml

# --- PROJECT PATHS ---
# Automatically locate the root of the project relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"        # API-fetched game logs (DVC-tracked)
ROSTERS_DATA_DIR = DATA_DIR / "rosters"    # API-fetched player metadata per season
CURATED_DATA_DIR = DATA_DIR / "curated"   # Hand-maintained reference files (proxies, targets)
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # ML pipeline outputs
SLATES_DATA_DIR = DATA_DIR / "slates"     # DraftKings/FanDuel contest salary files

# --- DATA INGESTION SETTINGS ---
# WNBA League ID is always '10' in nba_api
WNBA_LEAGUE_ID = '10'

# Active season for the dashboard and live optimizer
CURRENT_SEASON = '2025'

# Historical seasons for data pulls and backtesting
SEASONS_TO_FETCH = ['2021', '2022', '2023', '2024', '2025']

# --- PIPELINE CONTROL ---
# Set to True if you want to force a re-download of existing data
OVERWRITE = False

# API Retry Settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds

###################################################
# FEATURE ENGINEERING PARAMETERS
###################################################

# Scoring rulebook (see config/scoring/)
DEFAULT_SCORING_SYSTEM = 'wnba_default'

# Minimum number of games a player must play to be included
MIN_GAMES_THRESHOLD = 10 

# Feature Engineering: Rolling Average Windows
ROLLING_WINDOW_SHORT = 3
ROLLING_WINDOW_LONG = 10

# Columns that should NEVER be seen by the model (Metadata/Leakage)
META_FEATURES = [ # needed for testing/audit (but will be dropped before training)
    'GAME_DATE', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'SEASON'
]  
BAYESIAN_HELPERS = [
    'SEASON_GAME_NUM', 'PRIOR_SEASON_AVG', 'CURRENT_SEASON_AVG', 'NEW_SEASON_WEIGHT',
    'TEAM_GAME_NUM', 'PRIOR_WIN_PCT', 'CURRENT_WIN_PCT', 'TEAM_WEIGHT'
]
DROPPED_FEATURES = META_FEATURES + BAYESIAN_HELPERS + [
    'PLAYER_ID', 
    'TEAM_ID', 
    'GAME_ID',
    'season_id', 
    'SEASON_ID',
    'OPP_ABBREVIATION',
    'TEAM_NAME',
    'MATCHUP',
    'WL',
    'VIDEO_AVAILABLE',
    'SCRAPED_AT',
    'scraped_at',
    'salary',
    'POSITION',
    'match_name'
]

# The value we are trying to predict
TARGET_COL = 'FANTASY_PTS'



# --- SCORING CONFIGURATION --

SCORING_DIR = PROJECT_ROOT / "config" / "scoring"

# FANTASY GAME CONSTRAINTS (e.g., DraftKings WNBA Rules)
SALARY_CAP = 50000
ROSTER_SLOTS = {
    'G': 2,    # Guards
    'F': 3,    # Forwards
    'UTIL': 1  # Utility (Any position)
}
TOTAL_SLOTS = 6



#######################
## MODEL TRAINING #####
#######################

# --- MODEL HYPERPARAMETERS (Optimized via Grid Search 2026-03-03) ---
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 3,
    'n_estimators': 68,  # Using the 1.1x scaling rule from best_iteration 61 ((61 + 1) * 1.1)
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cuda'
}

# --- TRAINING SETTINGS ---
# Set to 1.0 for final production builds, or 0.8 for validation runs
TRAIN_SPLIT_PERCENT = 1.0
