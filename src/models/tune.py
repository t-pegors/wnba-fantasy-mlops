import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid
import mlflow
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv
import warnings
# Filter out the specific MLflow schema hint and the artifact_path deprecation
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", message="`artifact_path` is deprecated")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config

load_dotenv()

def tune_hyperparameters():
    print("🚀 Initiating Automated Grid Search...")

    # 1. Setup MLflow Tracking
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    mlflow.set_experiment("02_WNBA_Hyperparameter_Tuning")

    # 2. Load the Golden Table
    data_path = config.PROCESSED_DATA_DIR / "training_features.csv"
    df = pd.read_csv(data_path)
    
    # CRITICAL: Sort chronologically to perfectly match the baseline test set
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by='GAME_DATE')

    #3. Define Features (X) and Target (y)
    # We only drop the metadata, build_features handled the rest
    drop_cols = ['PLAYER_ID', 'GAME_DATE', 'SEASON', 'FANTASY_PTS']
    
    # Drop string columns and target, keep only features the model should see
    drop_cols = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=drop_cols).select_dtypes(include=['number'])
    y = df['FANTASY_PTS']

    # Ensure all features are floats to silence MLflow integer warnings
    X = X.astype('float64')

    # 4. Chronological Train-Test Split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"📊 Training on {len(X_train)} games, Testing on {len(X_test)} games.")

    # 5. Define the Search Grid (27 Combinations)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 200],
        'objective': ['reg:squarederror'],
        'random_state': [42]
    }
    
    grid = list(ParameterGrid(param_grid))
    print(f"🔬 Testing {len(grid)} different hyperparameter combinations...")

    best_mae = float('inf')
    best_params = None

    # 6. The Automated Tuning Loop
    for i, params in enumerate(grid):
        print(f"Testing combination {i+1}/{len(grid)}: {params}")
        
        with mlflow.start_run(run_name=f"grid_search_{i+1}"):
            mlflow.log_params(params)

            # Train the specific configuration
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            # Predict & Evaluate
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            
            # Log the results
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_rmse", rmse)
            mlflow.set_tag("model_type", "xgboost_tune")
            
            # log signatures and models
            signature = infer_signature(X_train, preds)
            mlflow.xgboost.log_model(model, "model", signature=signature)
            mlflow.set_tag("features_used", ", ".join(X_train.columns))

            # Track the reigning champion locally
            if mae < best_mae:
                best_mae = mae
                best_params = params

    print("-" * 30)
    print("🏆 GRID SEARCH COMPLETE!")
    print(f"✨ Best Target to Beat (Baseline): 6.71 MAE")
    print(f"🔥 Best XGBoost MAE: {best_mae:.2f}")
    print(f"🔧 Optimal Parameters: {best_params}")
    print("-" * 30)

if __name__ == "__main__":
    tune_hyperparameters()