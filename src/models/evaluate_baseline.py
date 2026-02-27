import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config

load_dotenv()

def evaluate_baselines():
    print("📊 Initiating Baseline Evaluation Pipeline...")

    # 1. Setup MLflow Tracking
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    # Create a dedicated experiment just for baselines
    mlflow.set_experiment("01_WNBA_Baselines")

    # 2. Load the Golden Table
    data_path = config.PROCESSED_DATA_DIR / "training_features.csv"
    df = pd.read_csv(data_path)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    # 3. Calculate the new baselines dynamically
    print("🧮 Calculating baseline predictions...")
    
    # Baseline 1: Last Game (Shift 1)
    df['PRED_LAST_GAME'] = df.groupby('PLAYER_ID')['FANTASY_PTS'].shift(1)
    
    # Baseline 2: Season-to-Date Average
    # Extract the year to group by season, then take an expanding mean
    df['SEASON'] = df['GAME_DATE'].dt.year
    df['PRED_SEASON_AVG'] = df.groupby(['PLAYER_ID', 'SEASON'])['FANTASY_PTS'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # Baseline 3: 3-Game Average (Already exists from build_features.py)
    # df['FPTS_3G_AVG'] 

    # 4. Clean and Split the Data
    # Drop rows where we couldn't calculate a baseline (e.g., the very first game of the season)
    df = df.dropna(subset=['PRED_LAST_GAME', 'PRED_SEASON_AVG', 'FPTS_3G_AVG', 'FANTASY_PTS'])

    # Sort chronologically by League Date, NOT by Player
    # This ensures our 80/20 split separates the past from the future for the entire league
    df = df.sort_values(by='GAME_DATE')
    
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    print(f"⏱️ Evaluating on {len(test_df)} test games (Chronological Split).")

    # 5. Define our baselines to loop through
    baselines = {
        "Baseline_Last_Game": "PRED_LAST_GAME",
        "Baseline_3G_Rolling": "FPTS_3G_AVG",
        "Baseline_Season_To_Date": "PRED_SEASON_AVG"
    }

    # 6. Evaluate and Log to DagsHub
    for run_name, col_name in baselines.items():
        with mlflow.start_run(run_name=run_name):
            
            mae = mean_absolute_error(test_df['FANTASY_PTS'], test_df[col_name])
            rmse = np.sqrt(mean_squared_error(test_df['FANTASY_PTS'], test_df[col_name]))
            
            print("-" * 30)
            print(f"🏆 {run_name}")
            print(f"📉 MAE:  {mae:.2f}")
            print(f"📉 RMSE: {rmse:.2f}")
            
            # Log metrics
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_rmse", rmse)
            
            # Tag it so it's easy to filter in the UI
            mlflow.set_tag("model_type", "baseline")

    print("-" * 30)
    print("✅ All baselines logged to DagsHub!")

if __name__ == "__main__":
    evaluate_baselines()