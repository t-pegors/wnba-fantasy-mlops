import os
import sys
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

# Path magic
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config
from src.utils.data_utils import normalize_name


def get_ncaa_fpts(player_name):
    """
    Scrapes Sports-Reference CBB for the player's final college season.
    Returns raw NCAA Fantasy Points (untranslated), or None if not found.
    """
    clean_url_name = player_name.lower().replace("'", "").replace(".", "").replace(" ", "-")
    url = f"https://www.sports-reference.com/cbb/players/{clean_url_name}-1.html"
    headers = {"User-Agent": "WNBA-MLOps-Pipeline/1.0"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'players_per_game'})
        if not table:
            return None

        df = pd.read_html(StringIO(str(table)))[0]
        df = df.dropna(subset=['Season'])
        final_season = df[df['Season'] != 'Career'].iloc[-1]

        pts   = float(final_season.get('PTS', 0))
        reb   = float(final_season.get('TRB', 0))
        ast   = float(final_season.get('AST', 0))
        stl   = float(final_season.get('STL', 0))
        blk   = float(final_season.get('BLK', 0))
        tov   = float(final_season.get('TOV', 0))
        threes = float(final_season.get('3P', 0))

        return (pts * 1.0) + (reb * 1.2) + (ast * 1.5) + (stl * 3.0) + (blk * 3.0) - (tov * 1.0) + (threes * 0.5)

    except Exception as e:
        print(f"      ⚠️ Scrape error for {player_name}: {e}")
        return None


def build_rookie_proxies():
    """
    Reads target_rookies.csv and produces rookie_proxies.csv in processed/.

    Routing logic:
      - ORIGIN_LEAGUE == 'NCAA': scrapes Sports-Reference, applies config.NCAA_ROOKIE_TAX
      - ORIGIN_LEAGUE == 'INTL': reads MANUAL_PROXY (raw overseas FPTS), applies config.INTL_PRO_TAX
    """
    print(f"\n{'='*55}")
    print("  Building Rookie Proxy Baselines")
    print(f"{'='*55}")

    target_path = config.CURATED_DATA_DIR / "target_rookies.csv"
    if not target_path.exists():
        print(f"❌ Target list not found at {target_path}")
        return

    targets = pd.read_csv(target_path)
    if 'PLAYER_NAME' not in targets.columns or 'ORIGIN_LEAGUE' not in targets.columns:
        print("❌ target_rookies.csv must have PLAYER_NAME and ORIGIN_LEAGUE columns.")
        return

    records = []

    for _, row in targets.iterrows():
        name = str(row['PLAYER_NAME']).strip()
        origin = str(row['ORIGIN_LEAGUE']).strip().upper()

        if origin == 'NCAA':
            print(f"   🎓 NCAA  | {name.title()} — scraping Sports-Reference...")
            raw_fpts = get_ncaa_fpts(name)

            if raw_fpts is not None:
                wnba_proxy = round(raw_fpts * config.NCAA_ROOKIE_TAX, 2)
                tier = 'ELITE' if wnba_proxy > 20 else 'ROTATION'
                print(f"      ✅ NCAA FPTS: {raw_fpts:.1f} → WNBA Proxy: {wnba_proxy} ({tier})")
                records.append({
                    'match_name': normalize_name(name),
                    'ORIGIN_LEAGUE': 'NCAA',
                    'RAW_FPTS': round(raw_fpts, 2),
                    'WNBA_ROOKIE_PROXY': wnba_proxy,
                    'PEDIGREE_TIER': tier,
                })
            else:
                print(f"      ⏭️  Not found. Check spelling or Sports-Reference URL.")

            time.sleep(3)  # Be polite to Sports-Reference servers

        elif origin == 'INTL':
            manual = row.get('MANUAL_PROXY', None)
            if pd.isna(manual) or str(manual).strip() == '':
                print(f"   🌍 INTL  | {name.title()} — ⚠️  MANUAL_PROXY is empty, skipping.")
                continue

            raw_fpts = float(manual)
            wnba_proxy = round(raw_fpts * config.INTL_PRO_TAX, 2)
            tier = 'ELITE' if wnba_proxy > 20 else 'ROTATION'
            print(f"   🌍 INTL  | {name.title()} — Raw FPTS: {raw_fpts:.1f} → WNBA Proxy: {wnba_proxy} ({tier})")
            records.append({
                'match_name': normalize_name(name),
                'ORIGIN_LEAGUE': 'INTL',
                'RAW_FPTS': raw_fpts,
                'WNBA_ROOKIE_PROXY': wnba_proxy,
                'PEDIGREE_TIER': tier,
            })

        else:
            print(f"   ⚠️  Unknown ORIGIN_LEAGUE '{origin}' for {name} — skipping.")

    if not records:
        print("\n💨 No proxy records generated.")
        return

    output_df = pd.DataFrame(records)
    output_path = config.PROCESSED_DATA_DIR / "rookie_proxies.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    ncaa_count = len(output_df[output_df['ORIGIN_LEAGUE'] == 'NCAA'])
    intl_count = len(output_df[output_df['ORIGIN_LEAGUE'] == 'INTL'])
    print(f"\n✅ Saved {len(output_df)} proxies → {output_path}")
    print(f"   NCAA: {ncaa_count}  |  INTL: {intl_count}")


if __name__ == "__main__":
    build_rookie_proxies()
