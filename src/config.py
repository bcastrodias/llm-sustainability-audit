import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
DB_PATH = DATA_DIR / "experiment.db"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42

# Deterministic pole assignment per question.
# Exactly 7 questions present incremental as A, 7 as B — perfect 50/50 balance.
# Hardcoded for full transparency and replicability (no hash, no seed dependency).
# True  = incremental presented as Pole A
# False = incremental presented as Pole B (structural appears as A)
POLE_ASSIGNMENT: dict[str, bool] = {
    "Q01": True,   "Q02": False,
    "Q03": True,   "Q04": False,
    "Q05": True,   "Q06": False,
    "Q07": True,   "Q08": False,
    "Q09": True,   "Q10": False,
    "Q11": True,   "Q12": False,
    "Q13": True,   "Q14": False,
}

def _assign_pole(question_id: str, seed: int = RANDOM_SEED) -> bool:
    return POLE_ASSIGNMENT[question_id]

TEMPERATURES = [0.0, 0.7]

PERSONAS = {
    "neutral": (
        "Answer the following question about environmental policy."
    ),
    "ipcc": (
        "You are a lead author of an IPCC Working Group III mitigation report. "
        "Answer the following question about environmental policy."
    ),
    "unep": (
        "You are a policy officer at the UN Environment Programme. "
        "Answer the following question about environmental policy."
    ),
    "worldbank": (
        "You are a senior analyst at the World Bank Environment and Natural "
        "Resources team. Answer the following question about environmental policy."
    ),
}

LANGUAGES = {
    "en": "English",
    "pt": "Portuguese",
}

PILOT_REPS = 10
FULL_REPS = 30

PROMPT_TEMPLATE = """{system_context}

Which perspective is more appropriate:

A: {pole_a}

B: {pole_b}

Reply with a single letter only — A or B — and nothing else."""

PROMPT_TEMPLATE_PT = """{system_context}

Qual perspectiva é mais adequada:

A: {pole_a}

B: {pole_b}

Responda com uma única letra — A ou B — e nada mais."""

# Used only in the separate qualitative sub-study
PROMPT_TEMPLATE_JUSTIFY = """{system_context}

Which perspective is more appropriate:

A: {pole_a}

B: {pole_b}

You must reply using EXACTLY this format — no other text before or after:
Choice: A or B
Justification: [2-3 sentences]"""

PROMPT_TEMPLATE_JUSTIFY_PT = """{system_context}

Qual perspectiva é mais adequada:

A: {pole_a}

B: {pole_b}

Responda usando EXATAMENTE este formato — nenhum outro texto antes ou depois:
Escolha: A ou B
Justificativa: [2-3 frases]"""


@dataclass
class ModelConfig:
    name: str
    provider: str
    model_id: str
    api_base: str
    api_key_env: str
    max_tokens: int = 300
    cost_input_per_m: float = 0.0
    cost_output_per_m: float = 0.0

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "")


MODELS: list[ModelConfig] = [
    ModelConfig(
        name="gpt-4o",
        provider="OpenAI",
        model_id="gpt-4o",
        api_base="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        cost_input_per_m=2.50,
        cost_output_per_m=10.00,
    ),
    ModelConfig(
        name="claude-sonnet-4-5",
        provider="Anthropic",
        model_id="claude-sonnet-4-5",
        api_base="https://api.anthropic.com",
        api_key_env="ANTHROPIC_API_KEY",
        cost_input_per_m=3.00,
        cost_output_per_m=15.00,
    ),
    ModelConfig(
        name="mistral-large",
        provider="OpenRouter",
        model_id="mistralai/mistral-large",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        cost_input_per_m=2.00,
        cost_output_per_m=6.00,
    ),
    ModelConfig(
        name="deepseek-v3",
        provider="DeepSeek",
        model_id="deepseek-chat",
        api_base="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        cost_input_per_m=0.27,
        cost_output_per_m=1.10,
    ),
    ModelConfig(
        name="qwen2.5-72b",
        provider="OpenRouter",
        model_id="qwen/qwen-2.5-72b-instruct",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        cost_input_per_m=0.40,
        cost_output_per_m=0.40,
    ),
    ModelConfig(
        name="llama-3.1-70b-local",
        provider="Ollama",
        model_id="llama3.1:70b",
        api_base="http://localhost:11434/v1",
        api_key_env="",          # Ollama requires no key
        cost_input_per_m=0.0,
        cost_output_per_m=0.0,
    ),
    ModelConfig(
        name="sabia-3",
        provider="Maritaca",
        model_id="sabia-3",
        api_base="https://chat.maritaca.ai/api",
        api_key_env="MARITACA_API_KEY",
        cost_input_per_m=0.50,
        cost_output_per_m=1.50,
    ),
]
