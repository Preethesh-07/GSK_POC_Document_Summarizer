"""
PDF document loader.

Uses PyPDFLoader to extract text from digital PDFs.
Returns a list of LangChain Document objects (one per page).
"""

from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PDFLoader:
    """Loads and extracts text from PDF documents using Unstructured API."""

    def load(self, pdf_path: str) -> list[Document]:
        """Load a PDF file and extract text semantically chunked by headers.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of Document objects, each representing a semantic chunk.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            Exception: If PDF parsing fails.
        """
        logger.info(f"Loading PDF with Unstructured API: {pdf_path}")
        
        loader = UnstructuredLoader(
            file_path=pdf_path,
            api_key=settings.unstructured_api_key,
            partition_via_api=True,
            strategy="fast",#hi_res, fast
            chunking_strategy="by_title",#by_title, by_page, by_title_with_content
            max_characters=settings.chunk_size,
            new_after_n_chars=settings.new_after_n_chars,
            combine_text_under_n_chars=settings.combine_under_n_chars,
            overlap=settings.chunk_overlap,
        )
        
        chunks: list[Document] = loader.load()
        logger.info(f"Loaded {len(chunks)} semantic chunks from {pdf_path}")
        return chunks
