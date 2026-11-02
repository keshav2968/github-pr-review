"""All settings in one place, read from environment variables or a local `.env` file.

Every field can be overridden with an env var of the same name in UPPER_CASE,
e.g. `MIN_CONFIDENCE=0.7`. Secrets use `SecretStr` so they never show up in logs
or tracebacks by accident.
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (any OpenAI-compatible endpoint; default = Gemini free tier) ---
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_api_key: SecretStr | None = None
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.2  # low = more consistent reviews between runs
    llm_timeout_s: float = 180
    llm_max_attempts: int = 6  # retries with exponential backoff on 429 / 5xx / timeouts
    llm_concurrency: int = 2  # parallel LLM calls (free tiers have low requests/minute)

    # --- Review tuning (the "precision knobs"; tune on the dev split, never on test) ---
    chunk_token_budget: int = 12_000  # max diff tokens per review call
    min_confidence: float = 0.6  # issues below this are not posted
    max_comments: int = 8  # cap per PR: a flood of comments gets ignored
    grounding_window: int = 3  # how far (lines) we search to fix an off-by-N line number
    max_file_context_chars: int = 40_000  # full-file context given to the verifier

    # --- State ---
    checkpoint_db: str = ".pr_review/checkpoints.sqlite"

    # --- GitHub ---
    github_repository: str | None = None  # "owner/repo"; None = use the repo of the current folder

    # --- Observability (read by the langsmith library itself) ---
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "pr-review"


@lru_cache
def get_settings() -> Settings:
    """Build settings once and reuse them (cheap to call anywhere)."""
    return Settings()
