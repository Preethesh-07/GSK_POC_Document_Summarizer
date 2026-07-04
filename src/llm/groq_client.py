"""
Groq LLM client factory.

Provides a configured ChatGroq instance.
All parameters are overridable for benchmarking different models.
"""

from langchain_groq import ChatGroq

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    max_retries: int | None = None,
) -> ChatGroq:
    """Create a configured ChatGroq LLM instance.

    Args:
        model: Model name override (default: settings.llm_model).
        temperature: Temperature override (default: settings.llm_temperature).
        max_retries: Max retries override (default: settings.llm_max_retries).

    Returns:
        Configured ChatGroq instance.
    """
    model_name = model or settings.llm_model
    temp = temperature if temperature is not None else settings.llm_temperature
    retries = max_retries or settings.llm_max_retries

    logger.debug(
        f"Initializing ChatGroq: model={model_name}, "
        f"temperature={temp}, max_retries={retries}"
    )

    return ChatGroq(
        model=model_name,
        api_key=settings.groq_api,
        temperature=temp,
        max_retries=retries,
    )
