"""
Unit tests for the DocumentChunker.
"""

from src.rag.chunker import DocumentChunker


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
    assert "token_estimate" in first
    assert "text" in first
    
    # Check that text is actually present
    assert len(first["text"]) > 0
    
    # Check that no chunk exceeds the max size significantly
    for chunk in chunks:
        assert len(chunk["text"]) <= 150
