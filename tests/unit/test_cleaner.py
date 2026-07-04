"""
Unit tests for the TextCleaner.
"""

from langchain_core.documents import Document

from src.rag.cleaner import TextCleaner


def test_cleaner_collapses_whitespace():
    """Test that multiple spaces and newlines are collapsed."""
    cleaner = TextCleaner()
    pages = [
        Document(
            page_content="This    is  a   \n\n\n\n test.\n\n\n",
            metadata={"page": 1}
        )
    ]
    
    cleaned = cleaner.clean(pages)
    
    assert "This is a \n\n test." in cleaned or "This is a \n\n test." == cleaned.strip()
    assert "    " not in cleaned
    assert "\n\n\n" not in cleaned


def test_cleaner_strips_non_printable():
    """Test that non-printable characters are removed."""
    cleaner = TextCleaner()
    # \x00 is a null byte (non-printable)
    pages = [
        Document(
            page_content="Hello\x00World",
            metadata={"page": 1}
        )
    ]
    
    cleaned = cleaner.clean(pages)
    
    assert "\x00" not in cleaned
    assert "HelloWorld" in cleaned


def test_cleaner_removes_repeated_headers():
    """Test that repeated headers across multiple pages are removed."""
    cleaner = TextCleaner()
    pages = [
        Document(page_content="CONFIDENTIAL REPORT\nPage content 1", metadata={"page": 1}),
        Document(page_content="CONFIDENTIAL REPORT\nPage content 2", metadata={"page": 2}),
        Document(page_content="CONFIDENTIAL REPORT\nPage content 3", metadata={"page": 3}),
        Document(page_content="CONFIDENTIAL REPORT\nPage content 4", metadata={"page": 4}),
    ]
    
    cleaned = cleaner.clean(pages)
    
    assert "CONFIDENTIAL REPORT" not in cleaned
    assert "Page content 1" in cleaned
    assert "Page content 4" in cleaned
