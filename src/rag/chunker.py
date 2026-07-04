"""
Document chunking using RecursiveCharacterTextSplitter.

Splits cleaned text into smaller chunks with configurable
size and overlap. Each chunk gets rich metadata.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """Splits cleaned document text into manageable chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, cleaned_text: str, document_id: str) -> list[dict]:
        """Split text into chunks with metadata.

        Args:
            cleaned_text: Cleaned document text.
            document_id: Unique identifier for the document.

        Returns:
            List of chunk dictionaries with:
                - id: Unique chunk identifier
                - index: Position in sequence
                - page: Source page (if available)
                - token_estimate: Approximate token count
                - text: Chunk text content
        """
        docs = self.splitter.create_documents([cleaned_text])

        chunks: list[dict] = []
        for i, doc in enumerate(docs):
            chunk = {
                "id": f"{document_id}_chunk_{i}",
                "index": i,
                "page": doc.metadata.get("page", -1),
                "token_estimate": len(doc.page_content) // 4,
                "text": doc.page_content,
            }
            chunks.append(chunk)

        logger.info(
            f"Created {len(chunks)} chunks from document '{document_id}' "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks
