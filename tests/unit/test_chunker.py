"""
Unit tests for the DocumentChunker.
"""
from unittest.mock import patch

from src.rag.chunker import DocumentChunker


def test_chunker_initialization_defaults():
    """Test if chunker uses default settings when initialized without parameters."""
    with patch("src.rag.chunker.settings") as mock_settings:
        mock_settings.chunk_size = 1000
        mock_settings.chunk_overlap = 200
        
        chunker = DocumentChunker()
        
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.splitter._chunk_size == 1000
        assert chunker.splitter._chunk_overlap == 200


def test_chunker_initialization_custom():
    """Test if custom values override default settings."""
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    
    assert chunker.chunk_size == 500
    assert chunker.chunk_overlap == 50
    assert chunker.splitter._chunk_size == 500
    assert chunker.splitter._chunk_overlap == 50


def test_chunker_creates_correct_chunks():
    """Test that chunking creates the right metadata and sizes."""
    # Create a small text that will be split
    text = "A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100
    
    # Tiny chunk size to force splits
    chunker = DocumentChunker(chunk_size=150, chunk_overlap=20)
    
    chunks = chunker.chunk(text, document_id="doc_123")
    
    assert len(chunks) > 1
    
    # Check first chunk structure
    first = chunks[0]
    assert "id" in first
    assert first["id"] == "doc_123_chunk_0"
    assert "index" in first
    assert first["index"] == 0
    assert "page" in first
    assert first["page"] == -1
    assert "token_estimate" in first
    assert "text" in first
    
    # Check that text is actually present
    assert len(first["text"]) > 0
    
    # Check that no chunk exceeds the max size significantly
    for chunk in chunks:
        assert len(chunk["text"]) <= 150


def test_chunker_small_text_no_split():
    """Test that small text doesn't get split unnecessarily."""
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)
    text = "This is a small text."
    
    chunks = chunker.chunk(text, document_id="doc_small")
    
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["id"] == "doc_small_chunk_0"


def test_chunker_empty_string():
    """Test chunking behavior with an empty string."""
    chunker = DocumentChunker(chunk_size=150, chunk_overlap=20)
    chunks = chunker.chunk("", document_id="doc_empty")
    
    # Langchain might return 0 chunks or 1 empty chunk depending on the version
    if len(chunks) > 0:
        assert len(chunks) == 1
        assert chunks[0]["text"] == ""
    else:
        assert len(chunks) == 0


def test_chunker_overlap_behavior():
    """Test that chunks have constraints on their length."""
    chunker = DocumentChunker(chunk_size=20, chunk_overlap=10)
    text = "word1 word2 word3 word4 word5 word6"
    
    chunks = chunker.chunk(text, document_id="doc_overlap")
    assert len(chunks) > 1
    
    for chunk in chunks:
        # Check that size is respected (approx)
        assert len(chunk["text"]) <= 20
