"""
Application settings loaded from environment variables.

All configuration is centralized here — no hardcoded values elsewhere.
Uses pydantic-settings to validate and load from .env file.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the document summarization system."""

    # ── API Keys ──────────────────────────────────────────────────────
    groq_api: str = Field(..., alias="GROQ_API")
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    postgres_url: str = Field(..., alias="POSTGRES_URL")
    unstructured_api_key: str = Field(default="", alias="UNSTRUCTURED_API_KEY")

    # ── LLM Config ────────────────────────────────────────────────────
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.3
    llm_max_retries: int = 3

    # ── Embedding Config ──────────────────────────────────────────────
    embedding_model: str = "gemini-embedding-001"

    # ── Chunking Config ───────────────────────────────────────────────
    chunk_size: int = 4000
    chunk_overlap: int = 400
    new_after_n_chars: int = 3800
    combine_under_n_chars: int = 1000

    # ── Summarization Config ──────────────────────────────────────────
    merge_batch_size: int = 5
    max_summary_tokens: int = 1500

    # ── Guardrails ────────────────────────────────────────────────────
    max_pages: int = 100
    max_file_size_mb: int = 50
    min_text_length: int = 100

    # ── Observability ─────────────────────────────────────────────────
    langsmith_api_key: str = Field(..., alias="LANGCHAIN_API_KEY")
    langsmith_project: str = Field(
        default="GSK document summarizer", alias="LANGCHAIN_PROJECT"
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton instance — import this everywhere
settings = Settings()
