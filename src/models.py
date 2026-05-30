import re
import time
from dataclasses import dataclass

import anthropic
from openai import OpenAI

from src.config import ModelConfig
from src.db import save_health


@dataclass
class Response:
    choice: str          # 'A', 'B', 'refused', or 'error'
    justification: str
    raw: str
    tokens_input: int
    tokens_output: int
    cost_usd: float
    latency_ms: int


def _parse_choice(raw: str, language: str) -> tuple[str, str]:
    """Extract choice letter (and optional justification) from model output."""
    # Primary: bare single letter (with optional trailing punctuation)
    bare = raw.strip().rstrip(".,:;!").strip().upper()
    if bare in ("A", "B"):
        return bare, ""

    # Fallback 1: structured format used in qualitative sub-study
    choice_key = "Escolha" if language == "pt" else "Choice"
    just_key = "Justificativa" if language == "pt" else "Justification"
    choice_match = re.search(rf"{choice_key}:\s*([AB])\b", raw, re.IGNORECASE)
    just_match = re.search(rf"{just_key}:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)

    if choice_match:
        letter = choice_match.group(1).upper()
        justification = just_match.group(1).strip() if just_match else ""
        return letter, justification

    # Fallback 2: letter after a colon or at end of string — "A resposta correta é: B"
    trailing = re.search(r"[:\s]([AB])\s*$", raw, re.IGNORECASE)
    if trailing:
        return trailing.group(1).upper(), raw.strip()

    # Fallback 3: unambiguous single letter in prose — "I choose A" (only one of A/B present)
    a_matches = len(re.findall(r"\bA\b", raw, re.IGNORECASE))
    b_matches = len(re.findall(r"\bB\b", raw, re.IGNORECASE))
    if a_matches > 0 and b_matches == 0:
        return "A", raw.strip()
    if b_matches > 0 and a_matches == 0:
        return "B", raw.strip()

    return "refused", raw.strip()


def _calc_cost(model: ModelConfig, tokens_in: int, tokens_out: int) -> float:
    return (
        tokens_in * model.cost_input_per_m / 1_000_000
        + tokens_out * model.cost_output_per_m / 1_000_000
    )


def call_openai_compat(
    model: ModelConfig,
    prompt: str,
    temperature: float,
    dry_run: bool = False,
) -> Response:
    """Handles OpenAI, DeepSeek, and OpenRouter (all share the OpenAI SDK interface)."""
    if dry_run:
        return Response("A", "dry-run justification", "dry-run", 10, 50, 0.0, 0)

    # Ollama doesn't require a real key; OpenAI SDK needs a non-empty string
    api_key = model.api_key or "ollama"
    client = OpenAI(api_key=api_key, base_url=model.api_base)
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model.model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=model.max_tokens,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    raw = resp.choices[0].message.content or ""
    tokens_in = resp.usage.prompt_tokens
    tokens_out = resp.usage.completion_tokens
    language = "pt" if "Escolha:" in prompt else "en"
    choice, justification = _parse_choice(raw, language)

    return Response(
        choice=choice,
        justification=justification,
        raw=raw,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        cost_usd=_calc_cost(model, tokens_in, tokens_out),
        latency_ms=latency_ms,
    )


def call_anthropic(
    model: ModelConfig,
    prompt: str,
    temperature: float,
    dry_run: bool = False,
) -> Response:
    if dry_run:
        return Response("A", "dry-run justification", "dry-run", 10, 50, 0.0, 0)

    client = anthropic.Anthropic(api_key=model.api_key)
    t0 = time.monotonic()
    resp = client.messages.create(
        model=model.model_id,
        max_tokens=model.max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    raw = resp.content[0].text if resp.content else ""
    tokens_in = resp.usage.input_tokens
    tokens_out = resp.usage.output_tokens
    language = "pt" if "Escolha:" in prompt else "en"
    choice, justification = _parse_choice(raw, language)

    return Response(
        choice=choice,
        justification=justification,
        raw=raw,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        cost_usd=_calc_cost(model, tokens_in, tokens_out),
        latency_ms=latency_ms,
    )


def call_model(model: ModelConfig, prompt: str, temperature: float, dry_run: bool = False) -> Response:
    if model.provider == "Anthropic":
        return call_anthropic(model, prompt, temperature, dry_run)
    return call_openai_compat(model, prompt, temperature, dry_run)


# --- Health checks ---

HEALTH_PROMPT = "Reply with exactly one word: OK"

def check_health(model: ModelConfig, dry_run: bool = False) -> bool:
    """Ping the model with a trivial prompt. Returns True if healthy."""
    if dry_run:
        print(f"  [dry-run] {model.name}: skipped")
        return True
    try:
        resp = call_model(model, HEALTH_PROMPT, temperature=0.0)
        ok = resp.choice not in ("error",)
        save_health(model.name, "ok" if ok else "error", resp.latency_ms, None)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {model.name} ({model.provider}) — {resp.latency_ms}ms")
        return ok
    except Exception as exc:
        save_health(model.name, "error", None, str(exc))
        print(f"  [FAIL] {model.name} ({model.provider}) — {exc}")
        return False


def check_all_models(models: list[ModelConfig], dry_run: bool = False) -> dict[str, bool]:
    print("\n-- API Health Check --")
    results = {}
    for m in models:
        if not m.api_key and m.provider != "Ollama":
            print(f"  [SKIP] {m.name}: no API key configured")
            results[m.name] = False
        else:
            results[m.name] = check_health(m, dry_run)
    healthy = sum(results.values())
    print(f"  {healthy}/{len(models)} models available\n")
    return results
