"""
PGVectorStore operations for storing and retrieving embeddings.

Uses langchain-postgres PGVectorStore (not the deprecated PGVector).
Tables are auto-created on first use — no manual migration needed.
"""

from langchain_core.documents import Document
from langchain_postgres import PGVectorStore

from src.config.settings import settings
from src.rag.embeddings import get_embedding_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Manages embedding storage and retrieval via PGVectorStore."""

    def __init__(self):
        # Convert connection string to psycopg3 format
        connection = settings.postgres_url
        if "postgresql://" in connection and "+psycopg" not in connection:
            connection = connection.replace(
                "postgresql://", "postgresql+psycopg://"
            )

        self._connection = connection
        self._embeddings = get_embedding_model()
        self._store: PGVectorStore | None = None

    @property
    def store(self) -> PGVectorStore:
        """Lazy initialization of the vector store."""
        if self._store is None:
            logger.info("Initializing PGVectorStore connection...")
            self._store = PGVectorStore(
                connection=self._connection,
                embeddings=self._embeddings,
                collection_name="document_chunks",
                use_jsonb=True,
            )
            logger.info("PGVectorStore initialized successfully")
        return self._store

    def store_chunks(
        self, chunks: list[dict], document_id: str
    ) -> list[str]:
        """Store document chunks with embeddings in pgvector.

        Args:
            chunks: List of chunk dictionaries from DocumentChunker.
            document_id: Unique document identifier.

        Returns:
            List of stored document IDs.
        """
        documents: list[Document] = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata={
                    "document_id": document_id,
                    "chunk_id": chunk["id"],
                    "chunk_index": chunk["index"],
                    "page_number": chunk["page"],
                    "token_estimate": chunk["token_estimate"],
                },
            )
            documents.append(doc)

        logger.info(
            f"Storing {len(documents)} chunks for document '{document_id}'..."
        )
        ids = self.store.add_documents(documents)
        logger.info(
            f"Successfully stored {len(ids)} chunks in pgvector"
        )
        return ids

    def similarity_search(
        self, query: str, k: int = 5
    ) -> list[Document]:
        """Search for similar chunks (for future Q&A use).

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            List of matching Document objects.
        """
        logger.info(f"Similarity search: '{query[:50]}...' (k={k})")
        results = self.store.similarity_search(query, k=k)
        logger.info(f"Found {len(results)} results")
        return results
