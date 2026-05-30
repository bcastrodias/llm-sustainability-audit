import concurrent.futures
import threading
import time

from src.config import (
    LANGUAGES, MODELS, PERSONAS, PROMPT_TEMPLATE, PROMPT_TEMPLATE_PT,
    TEMPERATURES, PILOT_REPS, FULL_REPS, ModelConfig,
)
from src.db import count_runs, get_all_questions, save_run
from src.models import Response, call_model


def _build_prompt(pole_a: str, pole_b: str, persona_key: str, language: str, order: str) -> str:
    template = PROMPT_TEMPLATE_PT if language == "pt" else PROMPT_TEMPLATE
    system_context = PERSONAS[persona_key]
    if order == "BA":
        pole_a, pole_b = pole_b, pole_a
    return template.format(system_context=system_context, pole_a=pole_a, pole_b=pole_b)


def _normalize_choice(raw_choice: str, order: str, incremental_is_a: bool) -> str:
    """
    Normalize the model's letter choice to the incremental/structural axis.
    Stored value: 'A' always means incremental, 'B' always means structural.
    Two things can invert the raw letter:
      - order BA: poles were swapped in the prompt
      - incremental_is_a=False: incremental was presented as B in the prompt
    An even number of inversions cancel out.
    """
    if raw_choice not in ("A", "B"):
        return raw_choice
    invert = (order == "BA") ^ (not incremental_is_a)
    if invert:
        return "B" if raw_choice == "A" else "A"
    return raw_choice


def _cells_for_model(model: ModelConfig, questions: list, n_reps: int) -> list[dict]:
    """Return all (question, order, persona, lang, temp) cells that still need runs."""
    cells = []
    for q in questions:
        for order in ("AB", "BA"):
            for persona_key in PERSONAS:
                for lang in LANGUAGES:
                    for temp in TEMPERATURES:
                        already = count_runs(
                            q["id"], model.name, order, persona_key, lang, temp
                        )
                        remaining = n_reps - already
                        if remaining > 0:
                            cells.append({
                                "q": q,
                                "order": order,
                                "persona_key": persona_key,
                                "lang": lang,
                                "temp": temp,
                                "start_rep": already + 1,
                                "remaining": remaining,
                            })
    return cells


def _run_model(
    model: ModelConfig,
    questions: list,
    n_reps: int,
    dry_run: bool,
    counter: list,
    counter_lock: threading.Lock,
    total: int,
    start_time: float,
) -> dict:
    """Run all pending cells for a single model sequentially. Designed to run in a thread."""
    done = errors = 0
    cells = _cells_for_model(model, questions, n_reps)

    for cell in cells:
        q = cell["q"]
        pole_a = q["pole_a_" + cell["lang"]]
        pole_b = q["pole_b_" + cell["lang"]]
        prompt = _build_prompt(pole_a, pole_b, cell["persona_key"], cell["lang"], cell["order"])

        for rep in range(cell["start_rep"], cell["start_rep"] + cell["remaining"]):
            run_data = {
                "question_id": q["id"],
                "model_name": model.name,
                "order_ab": cell["order"],
                "persona": cell["persona_key"],
                "language": cell["lang"],
                "temperature": cell["temp"],
                "rep_number": rep,
            }
            try:
                resp: Response = call_model(model, prompt, cell["temp"], dry_run=dry_run)
                run_data.update({
                    "choice": _normalize_choice(resp.choice, cell["order"], bool(q["incremental_is_a"])),
                    "justification": resp.justification,
                    "raw_response": resp.raw,
                    "tokens_input": resp.tokens_input,
                    "tokens_output": resp.tokens_output,
                    "cost_usd": resp.cost_usd,
                    "latency_ms": resp.latency_ms,
                })
                done += 1
            except Exception as exc:
                run_data.update({
                    "choice": "error",
                    "justification": str(exc),
                    "raw_response": str(exc),
                })
                errors += 1

            save_run(run_data)

            with counter_lock:
                counter[0] += 1
                completed = counter[0]

            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining_runs = total - completed
            eta_s = int(remaining_runs / rate) if rate > 0 else 0
            eta = f"{eta_s // 3600:02d}:{(eta_s % 3600) // 60:02d}:{eta_s % 60:02d}"

            label = (
                f"[{completed}/{total} {100*completed/total:5.1f}% ETA {eta}] "
                f"{q['id']} | {model.name:<22} | {cell['order']} | "
                f"{cell['persona_key']:<10} | {cell['lang']} | "
                f"t={cell['temp']} | rep {rep} -> {run_data.get('choice', '?')}"
            )
            print(label, flush=True)

            if not dry_run:
                time.sleep(0.3)

    return {"model": model.name, "done": done, "errors": errors}


def _estimate_total(questions, models, n_reps) -> int:
    return (
        len(questions) * len(models) * 2
        * len(PERSONAS) * len(LANGUAGES)
        * len(TEMPERATURES) * n_reps
    )


def run_experiment(
    n_reps: int | None = None,
    dry_run: bool = False,
    model_filter: list[str] | None = None,
    question_filter: list[str] | None = None,
) -> None:
    """
    Run all experiment cells in parallel across models.
    Each model runs in its own thread — questions are distributed evenly
    instead of exhausting one question before moving to the next.
    Safe to interrupt and resume: already-completed cells are skipped.
    """
    if n_reps is None:
        n_reps = PILOT_REPS

    questions = get_all_questions()
    if question_filter:
        questions = [q for q in questions if q["id"] in question_filter]

    models = MODELS
    if model_filter:
        models = [m for m in models if m.name in model_filter]

    # Count only truly pending runs (skips already-completed ones on resume)
    pending_per_model = {
        m.name: sum(c["remaining"] for c in _cells_for_model(m, questions, n_reps))
        for m in models
    }
    total_pending = sum(pending_per_model.values())
    total_possible = _estimate_total(questions, models, n_reps)
    already_done = total_possible - total_pending

    print(f"Experiment plan: {total_possible} total calls ({n_reps} reps/cell, {len(models)} models in parallel)")
    if already_done > 0:
        print(f"Resuming: {already_done} already completed, {total_pending} remaining")
    for name, pending in pending_per_model.items():
        print(f"  {name:<25} {pending:>5} pending")
    if dry_run:
        print("DRY RUN -- no real API calls")
    print()

    if total_pending == 0:
        print("Nothing to do — all cells already completed.")
        return

    counter = [0]
    counter_lock = threading.Lock()
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = {
            pool.submit(
                _run_model, m, questions, n_reps, dry_run,
                counter, counter_lock, total_pending, start_time
            ): m
            for m in models
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"\n-- {result['model']}: {result['done']} done, {result['errors']} errors")

    elapsed = time.time() - start_time
    print(f"\nExperiment complete in {elapsed/60:.1f} min.")
