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
  raw/                    # API-fetched WNBA game logs (DVC-tracked, stored in AWS S3)
  rosters/                # API-fetched player metadata per season (player_vault_*.csv)
  curated/                # Hand-maintained input files (target_rookies.csv — edit to add draft targets)
  processed/              # ML pipeline outputs (training_features.csv, rookie_proxies.csv)
  slates/                 # DraftKings/FanDuel contest salary CSVs (optimizer input)

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

- **Bayesian Cold-Start**: Rookies/international players blended from proxy data. NCAA targets scraped from Sports-Reference and translated via `NCAA_ROOKIE_TAX` (0.65x); INTL targets use hand-entered raw overseas FPTS translated via `INTL_PRO_TAX` (0.85x). Both tax rates live in `config.py`. Cascade fallback to flat 12.0 baseline. See `build_features.py` and `src/data/build_rookie_proxies.py`.
- **Rookie workflow**: Edit `data/curated/target_rookies.csv` (ORIGIN_LEAGUE = NCAA or INTL, fill MANUAL_PROXY for INTL), then run `python src/data/build_rookie_proxies.py` to regenerate `processed/rookie_proxies.csv`.
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
- `CURRENT_SEASON`: active season used by the dashboard and live optimizer
- `SEASONS_TO_FETCH`: historical seasons for data pulls and backtesting
- `NCAA_ROOKIE_TAX` / `INTL_PRO_TAX`: cold-start translation multipliers
- `ROLLING_WINDOW_SHORT` / `ROLLING_WINDOW_LONG`: feature window sizes (3 and 10 games)
- `SALARY_CAP`, `ROSTER_SLOTS`, position constraints: DraftKings lineup rules
- `DROPPED_FEATURES`: columns excluded from model input
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

## Seasonal Workflow (2026 Offseason Checklist)

### Now → NCAA Tournament (March)
- [ ] Finalize `data/curated/target_rookies.csv` with all 2026 draft prospects and known INTL targets
- [ ] Fill `MANUAL_PROXY` for any INTL players you have overseas stats for
- [ ] Confirm `CURRENT_SEASON = '2025'` and `SEASONS_TO_FETCH` are correct in `config.py`
- [ ] Run `python src/data/build_player_vault.py` if 2025 vault is incomplete

### After NCAA Tournament (early April — final college stats are locked)
- [ ] Run `python src/data/build_rookie_proxies.py` — scraper will now capture each NCAA target's **final season stats**
- [ ] Review terminal output; re-check any players that returned "Not found" (spelling/URL issues)
- [ ] Add/update any remaining INTL proxy values in `target_rookies.csv` and re-run

### After WNBA Draft (late April)
- [ ] Update `DRAFT_PICK` column in `target_rookies.csv` with actual draft positions
- [ ] Remove undrafted players from `target_rookies.csv` (or leave for reference — they won't match anyone)
- [ ] Add any surprise draftees who weren't on your radar
- [ ] Re-run `python src/data/build_rookie_proxies.py` for final proxy values

### Season Start (May)
- [ ] Update `CURRENT_SEASON = '2026'` in `config.py`
- [ ] Add `'2026'` to `SEASONS_TO_FETCH` in `config.py`
- [ ] Run `python src/data/build_player_vault.py` once initial rosters are posted
- [ ] Begin monitoring Model Health tab for drift as 2026 game logs accumulate
- [ ] Consider retraining model after ~3-4 weeks of 2026 data (enough to update priors)

### Mid-Season
- [ ] Re-run `python src/data/build_player_vault.py` after trade deadline (rosters change)
- [ ] Monitor the "⚠️ WARNING: players have NO historical or proxy data" alerts in `build_features.py` — these are players falling through to the flat 12.0 baseline who may need a proxy

---

## DVC & S3 Data Management

**Remote:** S3 bucket configured in `.dvc/config` as `myremote`.

### What's tracked and where

| File(s) | Tracking | Reason |
|---|---|---|
| `data/raw/wnba_*.csv` | DVC → S3 | Large game logs; slow/expensive to re-fetch |
| `data/rosters/player_vault_*.csv` | DVC → S3 | Rate-limited API; takes 5-10 min per season to rebuild |
| `data/processed/training_features.csv` | DVC → S3 | Versioned snapshot of Golden Table tied to each model |
| `data/curated/target_rookies.csv` | git | Hand-authored; small; you want full diff history |
| `data/processed/rookie_proxies.csv` | neither (gitignored) | Fast to regenerate from `target_rookies.csv` |
| `data/slates/` | neither (gitignored) | Ephemeral contest files; no value in versioning |

### Key commands

```bash
# Push local data changes → S3 (run after any data update)
dvc push

# Pull all DVC-tracked data from S3 (use on a fresh clone or new machine)
dvc pull

# Check what's out of sync before pushing
dvc status --cloud

# Add a new file to DVC tracking (e.g., a new season's vault)
dvc add data/rosters/player_vault_2026.csv
git add data/rosters/player_vault_2026.csv.dvc
dvc push
```

### When to run `dvc push`

- After daily ingest runs (game logs grow) — the GitHub Action updates the `.dvc` pointer but doesn't push the data
- After building a new player vault (`build_player_vault.py`)
- After retraining and regenerating `training_features.csv`

---

## Known Issues / Watch Out For

- **`.env` contains real credentials** — never commit it; it is gitignored.
- **No formal pytest suite** — validation is done via manual script execution and backtesting.
- `src/models/production/` directory is untracked — likely contains model artifacts that should be reviewed for DVC tracking.
