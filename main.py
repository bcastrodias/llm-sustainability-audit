"""
Entry point for the LLM environmental policy bias experiment.

Usage examples:
    python main.py health                          # check all APIs
    python main.py pilot                           # 10-rep pilot, all models
    python main.py pilot --dry-run                 # smoke test, no API calls
    python main.py pilot --models deepseek-v3      # pilot with one model
    python main.py full                            # 30-rep full experiment
    python main.py full --models gpt-4o claude-sonnet-4-5
    python main.py full --questions Q01 Q02 Q03    # specific questions only
"""

import argparse

from src.config import MODELS, PILOT_REPS, FULL_REPS
from src.db import init_db, seed_models
from src.experiment import run_experiment
from src.models import check_all_models


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Environmental Policy Bias Experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # health
    subparsers.add_parser("health", help="Ping all configured APIs")

    # pilot
    pilot_p = subparsers.add_parser("pilot", help=f"Run pilot ({PILOT_REPS} reps/cell)")
    pilot_p.add_argument("--dry-run", action="store_true", help="No real API calls")
    pilot_p.add_argument("--models", nargs="+", metavar="MODEL", help="Subset of models")
    pilot_p.add_argument("--questions", nargs="+", metavar="QID", help="Subset of question IDs")

    # full
    full_p = subparsers.add_parser("full", help=f"Run full experiment ({FULL_REPS} reps/cell)")
    full_p.add_argument("--dry-run", action="store_true")
    full_p.add_argument("--models", nargs="+", metavar="MODEL")
    full_p.add_argument("--questions", nargs="+", metavar="QID")

    args = parser.parse_args()

    # Always initialise DB first
    init_db()
    seed_models(MODELS)

    if args.command == "health":
        check_all_models(MODELS)

    elif args.command in ("pilot", "full"):
        n_reps = PILOT_REPS if args.command == "pilot" else FULL_REPS
        dry_run = getattr(args, "dry_run", False)
        model_filter = getattr(args, "models", None)
        question_filter = getattr(args, "questions", None)

        # Health check first — only run models that are alive
        health = check_all_models(MODELS, dry_run=dry_run)
        alive = [m for m in MODELS if health.get(m.name, False)]
        if model_filter:
            alive = [m for m in alive if m.name in model_filter]

        if not alive:
            print("No healthy models available. Aborting.")
            return

        run_experiment(
            n_reps=n_reps,
            dry_run=dry_run,
            model_filter=[m.name for m in alive],
            question_filter=question_filter,
        )


if __name__ == "__main__":
    main()
