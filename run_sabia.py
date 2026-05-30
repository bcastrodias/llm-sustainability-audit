"""
Run the pilot experiment for Sabia-3 only.
Resume-safe: already-completed runs are skipped automatically.

Usage:
    python run_sabia.py
"""
from src.config import MODELS, PILOT_REPS
from src.db import init_db, seed_models
from src.experiment import run_experiment
from src.models import check_health

if __name__ == "__main__":
    init_db()
    seed_models(MODELS)

    sabia = next(m for m in MODELS if m.name == "sabia-3")

    print("Checking Sabia-3 API...")
    if not check_health(sabia):
        print("API check failed. Aborting.")
        raise SystemExit(1)

    run_experiment(n_reps=PILOT_REPS, model_filter=["sabia-3"])
