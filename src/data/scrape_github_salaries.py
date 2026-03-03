import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Path magic
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config

def hunt_github_salaries():
    print("🕵️‍♀️ OSINT Protocol: Hunting for WNBA DraftKings Salaries...")
    
    # Securely load the GitHub token from the .env file
    load_dotenv(os.path.join(project_root, '.env'))
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not github_token:
        print("❌ CRITICAL: GITHUB_TOKEN not found in .env file.")
        print("Please create a .env file in the project root and add your token.")
        return

    # Inject the token into the HTTP Headers
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "WNBA-MLOps-Pipeline",
        "Authorization": f"token {github_token}"
    }
    
    # Search for files named DKSalaries.csv that contain the word WNBA
    queries = [
        "WNBA filename:DKSalaries.csv",
        "WNBA DraftKings filename:csv"
    ]
    
    # --- THESE WERE THE MISSING VARIABLES ---
    downloaded_dfs = []
    seen_urls = set()

    for query in queries:
        print(f"\n🔍 Executing Search Query: '{query}'")
        url = f"https://api.github.com/search/code?q={query}&per_page=10"
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 403:
            print("❌ GitHub Rate Limit Hit! (They block unauthenticated code searches quickly).")
            print("💡 Fix: Check that your token is valid and properly loaded.")
            return
        elif response.status_code != 200:
            print(f"⚠️ API Error {response.status_code}: {response.text}")
            continue
            
        items = response.json().get('items', [])
        print(f"📂 Found {len(items)} potential files in this sweep.")
        
        for item in items:
            raw_url = item['html_url'].replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            
            if raw_url in seen_urls:
                continue
            seen_urls.add(raw_url)
            
            print(f"⬇️ Downloading: {item['repository']['full_name']} / {item['name']}")
            
            try:
                # Fetch the raw CSV data
                csv_res = requests.get(raw_url)
                if csv_res.status_code == 200:
                    # Save temporarily to read with pandas
                    temp_path = "temp_salary.csv"
                    with open(temp_path, 'wb') as f:
                        f.write(csv_res.content)
                    
                    # Verify it's actually a DraftKings file (Checking for standard DK columns)
                    temp_df = pd.read_csv(temp_path, on_bad_lines='skip')
                    if 'Salary' in temp_df.columns and 'Name + ID' in temp_df.columns:
                        temp_df['source_repo'] = item['repository']['full_name']
                        downloaded_dfs.append(temp_df)
                        print("   ✅ Valid DraftKings file confirmed.")
                    else:
                        print("   ⏭️ False positive. Skipping.")
                        
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            except Exception as e:
                print(f"   ⚠️ Failed to parse: {e}")
                
            time.sleep(2) # Be polite to GitHub's servers

    if not downloaded_dfs:
        print("\n💨 Sweep complete. No valid salary files extracted.")
        return

    # Combine all found files into a master ledger
    print("\n🧬 Aggregating historical salary records...")
    master_df = pd.concat(downloaded_dfs, ignore_index=True)
    
    # Clean up the dataset to just what we need
    if 'Name' not in master_df.columns and 'Name + ID' in master_df.columns:
        # Extract just the name from "Name + ID" (e.g., "A'ja Wilson (1234567)")
        master_df['PLAYER_NAME'] = master_df['Name + ID'].str.extract(r'([A-Za-z\s\.\'\-]+)')
    elif 'Name' in master_df.columns:
        master_df['PLAYER_NAME'] = master_df['Name']
        
    master_df = master_df.rename(columns={'Salary': 'TRUE_SALARY'})
    
    # Keep the golden columns and drop duplicates
    cols_to_keep = ['PLAYER_NAME', 'TRUE_SALARY', 'Game Info', 'source_repo']
    master_df = master_df[[c for c in cols_to_keep if c in master_df.columns]].drop_duplicates()

    output_dir = config.DATA_DIR / "raw" / "salaries"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "scraped_dk_salaries.csv"
    
    master_df.to_csv(output_path, index=False)
    print(f"🎉 SUCCESS! Archived {len(master_df)} true salary records to {output_path}")

if __name__ == "__main__":
    hunt_github_salaries()