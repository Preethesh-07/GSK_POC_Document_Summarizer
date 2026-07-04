"""
PDF document loader.

Uses PyPDFLoader to extract text from digital PDFs.
Returns a list of LangChain Document objects (one per page).
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PDFLoader:
    """Loads and extracts text from PDF documents."""

    def load(self, pdf_path: str) -> list[Document]:
        """Load a PDF file and extract text from all pages.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of Document objects, one per page,
            each with page_content and metadata.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            Exception: If PDF parsing fails.
        """
        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages: list[Document] = loader.load()
        logger.info(f"Loaded {len(pages)} pages from {pdf_path}")
        return pages
