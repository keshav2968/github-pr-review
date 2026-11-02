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

    # --- GitHub ---
    github_repository: str | None = None  # "owner/repo"; None = use the repo of the current folder

    # --- Observability (read by the langsmith library itself) ---
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "pr-review"


@lru_cache
def get_settings() -> Settings:
    """Build settings once and reuse them (cheap to call anywhere)."""
    return Settings()
