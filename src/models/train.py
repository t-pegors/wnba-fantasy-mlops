import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# Path magic
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config

# Force load credentials
load_dotenv()

def train_model():
    print("🚀 Initiating XGBoost Model Training Pipeline...")

    # Setup MLflow Tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    db_user = os.getenv("MLFLOW_TRACKING_USERNAME")
    db_pass = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = db_user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = db_pass
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("WNBA_Fantasy_Predictor")

    # Load the Golden Table
    data_path = config.PROCESSED_DATA_DIR / "training_features.csv"
    df = pd.read_csv(data_path)
    
    # Define Features (X) and Target (y)
    target_col = 'FANTASY_PTS'
    
    ## Combined list of metadata to drop + the target itself
    # Use the centralized list from config, but only drop what's actually there
    to_drop = config.DROPPED_FEATURES + [config.TARGET_COL]
    actual_drops = [c for c in to_drop if c in df.columns]
    
    X = df.drop(columns=actual_drops).select_dtypes(include=['number'])
    y = df[config.TARGET_COL]

    # FAIL-SAFE: Force X to only keep numeric columns (integers and floats)
    X = X.select_dtypes(include=['number'])
    X = X.astype('float64')

    # 4. Data Split (Controlled by Config)
    if config.TRAIN_SPLIT_PERCENT < 1.0:
        split_idx = int(len(df) * config.TRAIN_SPLIT_PERCENT)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        print(f"📊 Validation Mode: Training on {len(X_train)} games, Testing on {len(X_test)}.")
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None
        print(f"📊 Production Mode: Training on ALL {len(X_train)} historical games.")

    # 6. Start the MLflow System of Record
    with mlflow.start_run(run_name="xgb_baseline_features"):
        # Log the hyperparameters
        mlflow.log_params(config.XGB_PARAMS)
        
        # Log the dataset size
        mlflow.log_param("train_size", len(X_train))
        
        print("🧠 Training XGBoost Regressor...")
        model = xgb.XGBRegressor(**config.XGB_PARAMS)
        model.fit(X_train, y_train)
        
        
        # 7. Evaluate Performance (ONLY if we have a test set)
        if X_test is not None:
            print("🔮 Generating Predictions for Validation...")
            predictions = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            
            print("-" * 30)
            print(f"📉 XGBoost MAE:  {mae:.2f} Fantasy Points")
            print(f"📉 XGBoost RMSE: {rmse:.2f} Fantasy Points")
            print("-" * 30)
            
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_rmse", rmse)
            
            # Save signatures based on the test predictions
            signature = infer_signature(X_train, predictions)
            mlflow.xgboost.log_model(model, "model", signature=signature)

            # The Gov-Grade Check
            if mae < 6.77:
                print("🏆 SUCCESS: Model beat the Naive Baseline (6.77)!")
            else:
                print("⚠️ WARNING: Model did not beat the Naive Baseline. Needs more feature engineering.")
                
        else:
            # In Production Mode, we log the model WITHOUT a signature/test-metrics
            print("🚀 Skipping Validation (Production Mode). Logging model...")
            mlflow.xgboost.log_model(model, "model")
        
        # Add a text tag so you can quickly read the features in the UI
        mlflow.set_tag("features_used", ", ".join(X_train.columns))
        
        

if __name__ == "__main__":
    train_model()