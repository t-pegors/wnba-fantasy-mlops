# src/utils/data_utils.py
import unicodedata

def normalize_name(name):
    if not isinstance(name, str): return ""
    # Lowercase, remove accents, and strip common punctuation
    name = name.lower().strip()
    name = "".join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.replace("'", "").replace("-", " ").replace(".", "")
    return name

# Add this to src/utils/data_utils.py
def normalize_position(pos):
    pos = str(pos).upper()
    if 'C' in pos:
        return pos.replace('C', 'F') # Centers are Forwards in WNBA DFS
    return pos