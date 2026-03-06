import os
from pathlib import Path
import yaml

# --- PROJECT PATHS ---
# Automatically locate the root of the project relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"        # API-fetched game logs (DVC-tracked)
ROSTERS_DATA_DIR = DATA_DIR / "rosters"    # API-fetched player metadata per season
CURATED_DATA_DIR = DATA_DIR / "curated"   # Hand-maintained reference files (target lists)

# Game day contest tracking (DVC-tracked → private S3; builds season-long evaluation dataset)
SLATE_SCORES_PATH = CURATED_DATA_DIR / "slate_scores.csv"   # One row per player per slate
CONTEST_LOG_PATH  = CURATED_DATA_DIR / "contest_log.csv"    # One row per contest (summary)
LAST_RETRAINED    = '2026-03-03'   # Update after each production model retrain
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # ML pipeline outputs
SLATES_DATA_DIR = DATA_DIR / "slates"     # DraftKings/FanDuel contest salary files

# Production model artifact
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "production" / "model.ubj"

# Holdout model artifact (train on all-but-one season; never overwrite production)
HOLDOUT_SEASON = '2025'        # Season to exclude from training and use as evaluation; set to None if not used
HOLDOUT_MODEL_DIR = PROJECT_ROOT / "src" / "models" / "holdout"
HOLDOUT_MODEL_PATH = HOLDOUT_MODEL_DIR / "model.ubj"

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

# Cold-Start / Rookie Translation Factors
NCAA_ROOKIE_TAX = 0.65   # NCAA → WNBA translation multiplier
INTL_PRO_TAX = 0.85      # International Pro → WNBA translation multiplier

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
# Legacy top-level values — kept for backward compatibility with bulk_backtest.py
SALARY_CAP = 50000
ROSTER_SLOTS = {
    'G': 2,    # Guards
    'F': 3,    # Forwards
    'UTIL': 1  # Utility (Any position)
}
TOTAL_SLOTS = 6

# --- MULTI-PLATFORM DFS CONFIGS ---
# Salary caps and roster slots for FanDuel/Yahoo are approximate.
# Verify from the first live slate CSV and update these values if needed.
PLATFORM_CONFIGS = {
    'draftkings': {
        'display_name': 'DraftKings',
        'salary_cap':   50000,
        'roster_slots': {'G': 2, 'F': 3, 'UTIL': 1},
        'total_slots':  6,
        'scoring_system': 'wnba_default',
        'csv_columns': {
            'name':      'Name',
            'salary':    'Salary',
            'position':  'Position',
            'team':      'TeamAbbrev',
            'game_info': 'Game Info',
            'injury':    None,           # DK omits scratched players from slate entirely
            'starting':  None,
        },
    },
    'fanduel': {
        'display_name': 'FanDuel',
        'salary_cap':   60000,   # ← verify from first FD slate
        'roster_slots': {'G': 2, 'F': 3, 'UTIL': 1},   # ← verify
        'total_slots':  6,
        'scoring_system': 'fanduel_wnba',
        'csv_columns': {
            'name':      'Nickname',   # FD exports use Nickname, not Name
            'salary':    'Salary',
            'position':  'Position',
            'team':      'Team',
            'game_info': 'Game',
            'injury':    'Injury Indicator',  # O / D / Q
            'starting':  None,
        },
    },
    'yahoo': {
        'display_name': 'Yahoo DFS',
        'salary_cap':   200,     # ← verify; Yahoo uses a credits system
        'roster_slots': {'G': 2, 'F': 3, 'UTIL': 1},   # ← verify
        'total_slots':  6,
        'scoring_system': 'yahoo_wnba',
        'csv_columns': {
            'name':      'Name',
            'salary':    'Salary',
            'position':  'Position',
            'team':      'Team',
            'game_info': 'Game',
            'injury':    'Injury Status',
            'starting':  'Starting',     # Yes / No — treat No as effectively Out
        },
    },
}



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
