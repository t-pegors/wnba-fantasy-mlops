import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv

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
    
    # Sort chronologically to match our baseline validation strategy
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by='GAME_DATE')
    
    # Define Features (X) and Target (y)
    target_col = 'FANTASY_PTS'
    
    # Only drop the metadata, build_features handled the rest
    drop_cols = ['PLAYER_ID', 'GAME_DATE', 'SEASON', 'FANTASY_PTS']
    
    # Ensure all drop columns actually exist in the dataframe before dropping
    drop_cols = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # FAIL-SAFE: Force X to only keep numeric columns (integers and floats)
    X = X.select_dtypes(include=['number'])

    # 4. Chronological Train-Test Split (80/20)
    # We don't use random split in sports, otherwise we'd predict yesterday's game using tomorrow's data!
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"📊 Training on {len(X_train)} games, Testing on {len(X_test)} games.")

    # 5. Define Model Hyperparameters
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.05,
        'max_depth': 5,
        'n_estimators': 100,
        'random_state': 42
    }

    # 6. Start the MLflow System of Record
    with mlflow.start_run(run_name="xgb_baseline_features"):
        # Log the hyperparameters
        mlflow.log_params(params)
        
        # Log the dataset size
        mlflow.log_param("train_size", len(X_train))
        
        print("🧠 Training XGBoost Regressor...")
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        print("🔮 Generating Predictions...")
        predictions = model.predict(X_test)
        
        # 7. Evaluate Performance
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        print("-" * 30)
        print(f"📉 XGBoost MAE:  {mae:.2f} Fantasy Points")
        print(f"📉 XGBoost RMSE: {rmse:.2f} Fantasy Points")
        print("-" * 30)
        
        # Log the metrics
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        
        # Save the feature signatures to DagsHub
        signature = infer_signature(X_train, predictions)
        mlflow.xgboost.log_model(model, "model", signature=signature)
        
        # Add a text tag so you can quickly read the features in the UI
        mlflow.set_tag("features_used", ", ".join(X_train.columns))
        
        # The Gov-Grade Check
        if mae < 6.77:
            print("🏆 SUCCESS: Model beat the Naive Baseline (6.77)!")
        else:
            print("⚠️ WARNING: Model did not beat the Naive Baseline. Needs more feature engineering.")

if __name__ == "__main__":
    train_model()