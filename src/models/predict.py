import os
import sys
import pandas as pd
import xgboost as xgb
from pathlib import Path

# Path magic
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config
from src.utils.data_utils import normalize_name

def run_inference(dk_df):
    """
    Takes a raw DraftKings slate dataframe, hydrates it with historical 
    features and OSINT proxies, and runs it through the XGBoost model.
    """
    print("🧠 Initiating XGBoost Inference Engine...")

    # Load the Production Model
    model_path = config.MODEL_PATH
    if not model_path.exists():
        print(f"❌ Error: No trained model found at {model_path}")
        # Fallback to dummy math so the UI doesn't crash during testing
        dk_df['Predicted_Pts'] = (dk_df['Salary'] / 1000) * 4.2 + (pd.Series(range(len(dk_df))) % 5)
        return dk_df

    # Load UBJ natively using XGBoost's scikit-learn wrapper for easy Pandas integration
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    # 2. Normalize the DraftKings Names for Matching
    dk_df['match_name'] = dk_df['Name'].apply(normalize_name)

    # 3. Load the Golden Table (To grab the most recent player anchors)
    golden_path = config.PROCESSED_DATA_DIR / "training_features.csv"
    if not golden_path.exists():
        print("❌ Error: Golden table not found.")
        dk_df['Predicted_Pts'] = 12.0
        return dk_df

    golden_df = pd.read_csv(golden_path)
    golden_df['match_name'] = golden_df['PLAYER_NAME'].apply(normalize_name)

    # Grab the absolute most recent row for each player to serve as their baseline
    latest_player_stats = golden_df.sort_values('GAME_DATE').groupby('match_name').tail(1)
    
    # Keep only the predictive features we need to carry forward
    # (In a true Day 1 scenario, we assume 7 days rest and no back-to-backs)
    anchor_columns = ['match_name', 'FPTS_SEASON_AVG', 'TEAM_WIN_PCT', 'OPP_WIN_PCT', 'WIN_PCT_DIFF']
    historical_anchors = latest_player_stats[anchor_columns]

    # 4. Hydrate the Slate
    inference_df = dk_df.merge(historical_anchors, on='match_name', how='left')

    # 5. Handle the Missing Rookies (The Cold Start Fallback)
    # If they weren't in the golden table, they get the 12.0 proxy 
    # (Assuming we already baked OSINT proxies into the golden table during build_features)
    inference_df['FPTS_SEASON_AVG'] = inference_df['FPTS_SEASON_AVG'].fillna(12.0)
    inference_df['TEAM_WIN_PCT'] = inference_df['TEAM_WIN_PCT'].fillna(0.500)
    inference_df['OPP_WIN_PCT'] = inference_df['OPP_WIN_PCT'].fillna(0.500)
    inference_df['WIN_PCT_DIFF'] = inference_df['TEAM_WIN_PCT'] - inference_df['OPP_WIN_PCT']

    # Inject static Opening Night assumptions
    inference_df['DAYS_REST'] = 7.0
    inference_df['IS_BACK_TO_BACK'] = 0
    inference_df['IS_HOME'] = 1 # Simplified: You can parse 'Game Info' to get actual Home/Away
    inference_df['FPTS_3G_AVG'] = inference_df['FPTS_SEASON_AVG']
    inference_df['FPTS_10G_AVG'] = inference_df['FPTS_SEASON_AVG']

    # 6. Format exactly for XGBoost (Matching the training columns)
    # Safely extract expected features from the UBJ artifact
    try:
        expected_features = model.feature_names_in_ 
    except AttributeError:
        # Fallback if saved via strict Booster API
        expected_features = model.get_booster().feature_names
    
    # Build the X matrix, filling any weird missing columns with 0
    X_predict = pd.DataFrame(index=inference_df.index)
    for col in expected_features:
        if col in inference_df.columns:
            X_predict[col] = inference_df[col]
        else:
            X_predict[col] = 0.0

    # 7. Execute the Prediction
    predictions = model.predict(X_predict)
    
    # 8. Attach back to the clean DraftKings dataframe
    dk_df['Predicted_Pts'] = predictions
    
    # Clean up
    dk_df = dk_df.drop(columns=['match_name'])
    
    print("✅ Inference complete. Lineups ready for optimization.")
    return dk_df