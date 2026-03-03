# src/utils/data_utils.py
import unicodedata
import yaml
from src import config

def normalize_name(name):
    if not isinstance(name, str): return ""
    # Lowercase, remove accents, and strip common punctuation
    name = name.lower().strip()
    name = "".join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.replace("'", "").replace("-", " ").replace(".", "")
    return name

def normalize_position(pos):
    """Converts WNBA API positions cleanly to DraftKings DFS slots."""
    pos = str(pos).upper().strip()
    
    # 1. Handle missing data safely to prevent pipeline crashes
    if pos in ['NAN', 'NONE', '']:
        return 'FORWARD' 
        
    # 2. Exorcise the ghost from the old CSVs
    pos = pos.replace('FENTER', 'FORWARD')
        
    # 3. Handle the full word FIRST before looking at single letters
    pos = pos.replace('CENTER', 'FORWARD')
    
    # 4. Standardize the abbreviations mapping Centers to Forwards
    if pos in ['C', 'F-C', 'C-F', 'F']:
        return 'FORWARD'
    elif pos == 'G':
        return 'GUARD'
        
    # 5. Leave combo strings intact (e.g., "GUARD-FORWARD") so the optimizer 
    # can trigger both the 'G' and 'F' eligibility checks!
    return pos


def load_scoring_system(system_name=None):
    """
    Loads a scoring configuration from the config/scoring directory.
    Usage: rules = load_scoring_system('fanduel_dfs')
    """
    # Fallback to the config default if nothing is passed
    if system_name is None:
        system_name = config.DEFAULT_SCORING_SYSTEM
        
    filepath = config.SCORING_DIR / f"{system_name}.yml"
    
    if not filepath.exists():
        raise FileNotFoundError(f"❌ Scoring system '{system_name}' not found at {filepath}")
        
    with open(filepath, 'r') as file:
        config_data = yaml.safe_load(file)
        
    print(f"Loaded Scoring System: {config_data['name']}")
    return config_data['weights']