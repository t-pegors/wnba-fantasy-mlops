import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from src import config
import os

def scrape_full_player_data():
    print("🚀 Initiating Deep Player Metadata Scrape...")
    url = "https://www.wnba.com/players"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Selecting the player containers
    player_cards = soup.select('.PlayerCard_pc__2_vS5') 
    all_player_data = []

    for card in player_cards:
        try:
            name = card.select_one('.PlayerCard_pcName__S3o_8').text.strip()
            # The bio line usually looks like: "#22 • Guard • Indiana Fever • Height 6-0 • Exp 1 yr"
            bio_text = card.select_one('.PlayerCard_pcBio__3_vS5').text.strip()
            
            # Use Regex to extract specific patterns
            height_match = re.search(r'Height (\d-\d+)', bio_text)
            exp_match = re.search(r'Exp (\d+|Rookie)', bio_text)
            pos_match = re.search(r'• ([a-zA-Z-]+) •', bio_text)
            
            # Convert Height (6-2) to total inches (74)
            height_inches = None
            if height_match:
                h = height_match.group(1).split('-')
                height_inches = int(h[0]) * 12 + int(h[1])
            
            # Convert Experience to an integer
            exp_years = 0
            if exp_match:
                val = exp_match.group(1)
                exp_years = 0 if val == 'Rookie' else int(val)

            all_player_data.append({
                'PLAYER_NAME': name,
                'PRIMARY_POS': pos_match.group(1)[0] if pos_match else 'U',
                'HEIGHT_INCHES': height_inches,
                'EXP_YEARS': exp_years,
                'BIO_STRING': bio_text
            })
        except Exception as e:
            continue

    df = pd.DataFrame(all_player_data)
    
    # Save to our metadata store
    output_path = config.DATA_DIR / "metadata" / "player_deep_stats.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Deep Metadata Scrape Complete. Captured {len(df)} players.")
    return df

if __name__ == "__main__":
    scrape_full_player_data()