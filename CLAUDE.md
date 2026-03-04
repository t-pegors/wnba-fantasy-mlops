# CLAUDE.md — WNBA Fantasy MLOps Engine

## Project Purpose

Production-grade MLOps pipeline for WNBA DFS (Daily Fantasy Sports) lineup optimization on DraftKings. The system:

1. Ingests daily WNBA game statistics (nba_api)
2. Engineers features with Bayesian cold-start logic for rookies
3. Trains/tunes XGBoost regression models to predict fantasy points
4. Solves a constrained optimization problem (PuLP) to build optimal lineups
5. Serves results via a Streamlit dashboard

---

## Repository Structure

```
src/
  config.py               # Central configuration — paths, hyperparams, DFS constraints, scoring
  data/                   # Data ingestion scripts (loaders + scrapers)
  features/
    build_features.py     # Feature engineering pipeline ("Golden Table" output)
  models/
    train.py              # XGBoost training with MLflow tracking
    tune.py               # Hyperparameter grid search
    predict.py            # Inference engine
    analyze_features.py   # Feature importance
    evaluate_baseline.py  # Baseline comparison
  inference/
    optimizer.py          # Linear programming lineup optimizer (PuLP)
    backtest.py           # Single-date historical simulation
    bulk_backtest.py      # Batch backtest evaluation
  diagnostics/            # MLflow connectivity + model registry utilities
  utils/
    data_utils.py         # Name normalization, position mapping, scoring loader

app/frontend/
  dashboard.py            # Streamlit UI (Daily Optimizer / Data Vault / Model Health tabs)

config/scoring/
  wnba_default.yml        # Fantasy point weights (configurable without code changes)
  nba_default.yml         # NBA alternative scoring

data/
  raw/                    # Raw game logs (DVC-tracked, stored in AWS S3)
  processed/              # Engineered features (training_features.csv)
  metadata/               # OSINT proxy data (NCAA/international pedigree)

.github/workflows/
  daily_ingest.yml        # Automated daily ingestion at 2 AM EST

docs/architecture/
  ADR-001-initial-stack.md
```

---

## Data Pipeline (End-to-End)

```
GitHub Actions (2 AM EST)
  → Data Ingestion (wnba_loader.py, scrapers)
  → DVC version + push to AWS S3
  → Feature Engineering (build_features.py → training_features.csv)
  → Model Training (train.py → MLflow/DagsHub)
  → Inference (predict.py + optimizer.py → optimal lineup)
  → Dashboard (dashboard.py)
```

---

## Key Design Decisions

- **Bayesian Cold-Start**: Rookies/international players blended from NCAA (0.65x) or INTL (0.85x) proxy data. Cascade fallback to flat 12.0 baseline. See `build_features.py:112-184`.
- **No data leakage**: Raw box score stats (PTS, REB, etc.) are dropped before training; only engineered features used.
- **Temporal validation**: Data sorted chronologically; no future information leaks into past predictions.
- **Scoring abstraction**: Fantasy point weights live in `config/scoring/*.yml` — change scoring systems without touching code.
- **DVC + S3**: Raw data files are not in git. Only `.dvc` pointer files are committed. Run `dvc pull` to hydrate data locally.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| ML Model | XGBoost (GPU via CUDA) |
| Optimization | PuLP (CBC solver) |
| MLOps | MLflow + DagsHub |
| Data versioning | DVC + AWS S3 |
| Automation | GitHub Actions |
| Dashboard | Streamlit |
| Data source | nba_api, requests/BeautifulSoup |

---

## Configuration

All key settings live in `src/config.py`:
- `SEASONS`: which WNBA seasons to ingest
- `ROLLING_WINDOW_SHORT` / `ROLLING_WINDOW_LONG`: feature window sizes (3 and 10 games)
- `SALARY_CAP`, `ROSTER_SLOTS`, position constraints: DraftKings lineup rules
- `FEATURES_TO_DROP`: columns excluded from model input
- XGBoost hyperparameters (tuned via `tune.py`)

---

## Environment & Credentials

Credentials are stored in `.env` (not committed). Required variables:
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` — S3 access for DVC
- `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` — DagsHub
- `GITHUB_TOKEN` — used by daily ingest workflow

For CI/CD, these are injected via GitHub Secrets.

---

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Pull versioned data from S3
dvc pull

# Build features
python src/features/build_features.py

# Train model
python src/models/train.py

# Launch dashboard
streamlit run app/frontend/dashboard.py
```

---

## Known Issues / Watch Out For

- **`.env` contains real credentials** — never commit it; it is gitignored.
- **No formal pytest suite** — validation is done via manual script execution and backtesting.
- `src/models/production/` directory is untracked (listed in git status as `??`) — likely contains model artifacts that should be reviewed for DVC tracking.
- `src/config.py`, `src/models/train.py`, and `src/models/tune.py` have uncommitted modifications — review before committing.
