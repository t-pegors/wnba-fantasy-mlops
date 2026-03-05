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

def train_model(holdout_season=None):
    """
    Train the XGBoost fantasy points predictor.

    Args:
        holdout_season (str | None): e.g. '2025'. When provided, that season is
            excluded from training and used as an evaluation holdout. The model
            is saved to config.HOLDOUT_MODEL_PATH — NOT the production directory.
            When None, all data is used and the model is saved to config.MODEL_PATH.
    """
    mode = f"HOLDOUT ({holdout_season})" if holdout_season else "PRODUCTION (all seasons)"
    print(f"🚀 Initiating XGBoost Training Pipeline — Mode: {mode}")

    # Setup MLflow Tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    db_user = os.getenv("MLFLOW_TRACKING_USERNAME")
    db_pass = os.getenv("MLFLOW_TRACKING_PASSWORD")

    os.environ["MLFLOW_TRACKING_USERNAME"] = db_user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = db_pass
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "WNBA_Fantasy_Holdout" if holdout_season else "WNBA_Fantasy_Predictor"
    mlflow.set_experiment(experiment_name)

    # Load the Golden Table
    data_path = config.PROCESSED_DATA_DIR / "training_features.csv"
    df = pd.read_csv(data_path)

    # --- Split by holdout season if requested ---
    if holdout_season:
        if 'SEASON' not in df.columns:
            print(f"❌ 'SEASON' column not found in training data. Cannot create holdout split.")
            return

        holdout_season_val = int(holdout_season)
        train_df = df[df['SEASON'] != holdout_season_val].copy()
        holdout_df = df[df['SEASON'] == holdout_season_val].copy()

        if train_df.empty:
            print(f"❌ No training rows remain after excluding season {holdout_season}.")
            return
        if holdout_df.empty:
            print(f"❌ No rows found for holdout season {holdout_season}.")
            return

        print(f"📊 Holdout Mode: Training on {len(train_df)} rows, evaluating on {len(holdout_df)} rows ({holdout_season}).")
    else:
        train_df = df
        holdout_df = None
        print(f"📊 Production Mode: Training on ALL {len(train_df)} historical rows.")

    # Build feature matrices
    to_drop = config.DROPPED_FEATURES + [config.TARGET_COL]
    actual_drops = [c for c in to_drop if c in train_df.columns]

    X_train = train_df.drop(columns=actual_drops).select_dtypes(include=['number']).astype('float64')
    y_train = train_df[config.TARGET_COL]

    if holdout_df is not None:
        holdout_actual_drops = [c for c in to_drop if c in holdout_df.columns]
        X_holdout = holdout_df.drop(columns=holdout_actual_drops).select_dtypes(include=['number']).astype('float64')
        # Align columns to training set
        X_holdout = X_holdout.reindex(columns=X_train.columns, fill_value=0.0)
        y_holdout = holdout_df[config.TARGET_COL]

    run_name = f"xgb_holdout_{holdout_season}" if holdout_season else "xgb_production"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config.XGB_PARAMS)
        mlflow.log_param("train_size", len(X_train))
        if holdout_season:
            mlflow.log_param("holdout_season", holdout_season)
            mlflow.log_param("holdout_size", len(X_holdout))

        print("🧠 Training XGBoost Regressor...")
        model = xgb.XGBRegressor(**config.XGB_PARAMS)
        model.fit(X_train, y_train)

        # Evaluate on the holdout season
        if holdout_df is not None:
            print(f"🔮 Evaluating on holdout season {holdout_season}...")
            predictions = model.predict(X_holdout)

            mae = mean_absolute_error(y_holdout, predictions)
            rmse = np.sqrt(mean_squared_error(y_holdout, predictions))

            print("-" * 40)
            print(f"📉 Holdout MAE  ({holdout_season}): {mae:.2f} Fantasy Points")
            print(f"📉 Holdout RMSE ({holdout_season}): {rmse:.2f} Fantasy Points")
            print("-" * 40)

            mlflow.log_metric("holdout_mae", mae)
            mlflow.log_metric("holdout_rmse", rmse)

            if mae < 6.77:
                print("🏆 SUCCESS: Holdout model beat the Naive Baseline (6.77)!")
            else:
                print("⚠️  WARNING: Holdout model did not beat the Naive Baseline.")

            signature = infer_signature(X_train, predictions)
            mlflow.xgboost.log_model(model, "model", signature=signature)

            # Save locally to holdout directory (NOT production)
            config.HOLDOUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_model(config.HOLDOUT_MODEL_PATH)
            print(f"💾 Holdout model saved to: {config.HOLDOUT_MODEL_PATH}")

        else:
            # Production mode — no test set, log without metrics
            print("🚀 Skipping validation (Production Mode). Logging model...")
            mlflow.xgboost.log_model(model, "model")

            # Save locally to production directory
            config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(config.MODEL_PATH)
            print(f"💾 Production model saved to: {config.MODEL_PATH}")

        mlflow.set_tag("features_used", ", ".join(X_train.columns))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train WNBA XGBoost model.")
    parser.add_argument(
        "--holdout-season",
        type=str,
        default=None,
        help="Season to hold out for evaluation (e.g. '2025'). Omit for production mode."
    )
    args = parser.parse_args()
    # Fall back to config if flag not provided on the CLI
    holdout_season = args.holdout_season if args.holdout_season is not None else config.HOLDOUT_SEASON
    train_model(holdout_season=holdout_season)
