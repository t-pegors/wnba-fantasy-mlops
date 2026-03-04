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
import time
import cupy as cp
import warnings
# Filter out the specific MLflow schema hint and the artifact_path deprecation
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", message="`artifact_path` is deprecated")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config

load_dotenv()

def tune_hyperparameters(holdout_season=None):
    """
    Run a grid search over XGB_PARAMS combinations.

    Args:
        holdout_season (str | None): e.g. '2025'. When provided, that season is
            excluded from training and used as the evaluation set. Runs log to
            the '02_WNBA_Hyperparameter_Tuning_Holdout' experiment.
            When None, uses the standard 80/20 chronological split and logs to
            '02_WNBA_Hyperparameter_Tuning'.
    """
    mode = f"HOLDOUT ({holdout_season})" if holdout_season else "80/20 CHRONOLOGICAL"
    print(f"🚀 Initiating Automated Grid Search — Split: {mode}")

    # 1. Setup MLflow Tracking
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    experiment_name = (
        "02_WNBA_Hyperparameter_Tuning_Holdout"
        if holdout_season
        else "02_WNBA_Hyperparameter_Tuning"
    )
    mlflow.set_experiment(experiment_name)

    # 2. Load the Golden Table
    data_path = config.PROCESSED_DATA_DIR / "training_features.csv"
    df = pd.read_csv(data_path)

    to_drop = config.DROPPED_FEATURES + [config.TARGET_COL]
    actual_drops = [c for c in to_drop if c in df.columns]

    X = df.drop(columns=actual_drops).select_dtypes(include=['number']).astype('float64')
    y = df[config.TARGET_COL]

    # Save feature names before converting to CuPy (arrays have no .columns)
    feature_names = list(X.columns)

    # 3. Train / Test Split
    if holdout_season:
        if 'SEASON' not in df.columns:
            print("❌ 'SEASON' column not found. Cannot create holdout split.")
            return

        train_mask = df['SEASON'] != holdout_season
        test_mask  = df['SEASON'] == holdout_season

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f"❌ Season split produced an empty set. Check holdout_season='{holdout_season}'.")
            return

        X_train_pd = X[train_mask]
        X_test_pd  = X[test_mask]
        y_train_pd = y[train_mask]
        y_test_pd  = y[test_mask]
        print(f"📊 Holdout split: {len(X_train_pd)} train rows, {len(X_test_pd)} holdout rows ({holdout_season}).")
    else:
        split_idx  = int(len(df) * 0.8)
        X_train_pd = X.iloc[:split_idx]
        X_test_pd  = X.iloc[split_idx:]
        y_train_pd = y.iloc[:split_idx]
        y_test_pd  = y.iloc[split_idx:]
        print(f"📊 80/20 split: {len(X_train_pd)} train rows, {len(X_test_pd)} test rows.")

    # Move to GPU
    X_train = cp.asarray(X_train_pd.values)
    X_test  = cp.asarray(X_test_pd.values)
    y_train = cp.asarray(y_train_pd.values)
    y_test  = cp.asarray(y_test_pd.values)

    # 4. Define the Search Grid
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'objective': ['reg:squarederror'],
        'random_state': [42],
        'n_estimators': [1000]  # High ceiling; early stopping will handle the rest
    }

    grid = list(ParameterGrid(param_grid))
    print(f"🔬 Testing {len(grid)} different hyperparameter combinations...")

    best_mae = float('inf')
    best_params = None

    # 5. The Automated Tuning Loop
    for i, params in enumerate(grid):

        params.update({'tree_method': 'hist', 'device': 'cuda'})

        start_time = time.time()
        print(f"🚀 [{i+1}/{len(grid)}] Training with {params}...")

        with mlflow.start_run(run_name=f"grid_search_{i+1}"):
            mlflow.log_params(params)
            if holdout_season:
                mlflow.log_param("holdout_season", holdout_season)

            model = xgb.XGBRegressor(
                **params,
                early_stopping_rounds=10
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

            preds = model.predict(X_test)

            mae      = mean_absolute_error(y_test, preds)
            rmse     = np.sqrt(mean_squared_error(y_test, preds))
            duration = time.time() - start_time

            print(f"✅ Run {i+1} complete in {duration:.2f}s | MAE: {mae:.4f}")
            mlflow.log_metric("duration_seconds", duration)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_rmse", rmse)
            mlflow.set_tag("model_type", "xgboost_tune")
            mlflow.set_tag("best_iteration", model.best_iteration)

            signature = infer_signature(X_train, preds)
            mlflow.xgboost.log_model(model, name="model", signature=signature)
            mlflow.set_tag("features_used", ", ".join(feature_names))

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
    import argparse
    parser = argparse.ArgumentParser(description="Grid search for WNBA XGBoost hyperparameters.")
    parser.add_argument(
        "--holdout-season",
        type=str,
        default=None,
        help="Season to hold out as evaluation set (e.g. '2025'). Omit for 80/20 chronological split."
    )
    args = parser.parse_args()
    # Fall back to config if flag not provided on the CLI
    holdout_season = args.holdout_season if args.holdout_season is not None else config.HOLDOUT_SEASON
    tune_hyperparameters(holdout_season=holdout_season)
