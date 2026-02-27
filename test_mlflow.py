import os
import sys
import mlflow
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def test_connection():
    # 1. Set up the connection
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("❌ MLFLOW_TRACKING_URI not found in .env")
        
    mlflow.set_tracking_uri(tracking_uri)
    print(f"🔗 Connected to Tracking Server: {tracking_uri}")

    # 2. Set the active experiment
    mlflow.set_experiment("00_Connection_Test")

    # 3. Start a run and log some dummy data
    print("🚀 Starting dummy run...")
    with mlflow.start_run(run_name="ping_test"):
        # Log a fake hyperparameter
        mlflow.log_param("test_parameter", 42)
        
        # Log a fake performance metric
        mlflow.log_metric("dummy_mae", 6.5)
        
        print("✅ Successfully logged parameters and metrics to DagsHub!")

if __name__ == "__main__":
    test_connection()