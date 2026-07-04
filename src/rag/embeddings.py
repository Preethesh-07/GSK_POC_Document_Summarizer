"""
Gemini embedding model wrapper.

Uses GoogleGenerativeAIEmbeddings from langchain-google-genai.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Create a configured Gemini embedding model instance.

    Returns:
        GoogleGenerativeAIEmbeddings configured with the API key
        and model from settings.
    """
    logger.debug(f"Initializing embedding model: {settings.embedding_model}")
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
    )
