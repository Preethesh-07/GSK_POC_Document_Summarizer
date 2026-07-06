"""
Unit tests for the TextCleaner.
"""

from langchain_core.documents import Document

from src.rag.cleaner import TextCleaner


def test_cleaner_empty_pages():
    """Test that an empty list of pages returns an empty string."""
    cleaner = TextCleaner()
    assert cleaner.clean([]) == ""


def test_cleaner_collapses_whitespace():
    """Test that multiple spaces and newlines are collapsed properly."""
    cleaner = TextCleaner()
    pages = [
        Document(
            page_content="This    is  a   \n\n\n\n test.\n\n\nAnd   more   spacing.",
            metadata={"page": 1}
        )
    ]
    
    cleaned = cleaner.clean(pages)
    
    # \n\n\n\n -> \n\n, and line strip
    # multiple spaces -> single space
    assert "This is a" in cleaned
    assert "\n\ntest." in cleaned
    assert "\n\nAnd more spacing." in cleaned
    assert "    " not in cleaned


def test_cleaner_strips_non_printable():
    """Test that non-printable characters are removed."""
    cleaner = TextCleaner()
    # \x00 is a null byte, \x0b is vertical tab (which isn't in standard whitespace allowed)
    pages = [
        Document(
            page_content="Hello\x00World\x0b!",
            metadata={"page": 1}
        )
    ]
    
    cleaned = cleaner.clean(pages)
    
    assert "\x00" not in cleaned
    assert "\x0b" not in cleaned
    # The \x0b vertical tab is matched by [^\S\n] in whitespace normalization
    # and replaced with a single space before non-printables are stripped.
    assert "HelloWorld !" in cleaned


def test_cleaner_joins_pages_correctly():
    """Test that multiple pages are joined with double newlines."""
    cleaner = TextCleaner()
    # 2 pages means header removal doesn't run (< 3 pages limit)
    pages = [
        Document(page_content="Page 1 text.", metadata={"page": 1}),
        Document(page_content="Page 2 text.", metadata={"page": 2}),
    ]
    
    cleaned = cleaner.clean(pages)
    assert cleaned == "Page 1 text.\n\nPage 2 text."


def test_cleaner_removes_repeated_headers_and_footers():
    """Test that repeated headers/footers across multiple pages are removed."""
    cleaner = TextCleaner()
    # Threshold is 70%, so in 4 pages, it must appear on at least 3 pages.
    pages = [
        Document(page_content="CONFIDENTIAL\nContent 1\nFOOTER_TEXT", metadata={"page": 1}),
        Document(page_content="CONFIDENTIAL\nContent 2\nFOOTER_TEXT", metadata={"page": 2}),
        Document(page_content="CONFIDENTIAL\nContent 3\nFOOTER_TEXT", metadata={"page": 3}),
        Document(page_content="CONFIDENTIAL\nContent 4\nFOOTER_TEXT", metadata={"page": 4}),
    ]
    
    cleaned = cleaner.clean(pages)
    
    assert "CONFIDENTIAL" not in cleaned
    assert "FOOTER_TEXT" not in cleaned
    assert "Content 1" in cleaned
    assert "Content 4" in cleaned


def test_cleaner_keeps_infrequent_headers():
    """Test that headers not meeting the repetition threshold are kept."""
    cleaner = TextCleaner()
    # In 4 pages, threshold is 2.8. So 2 occurrences should NOT be removed.
    pages = [
        Document(page_content="RARE HEADER\nContent A", metadata={"page": 1}),
        Document(page_content="RARE HEADER\nContent B", metadata={"page": 2}),
        Document(page_content="NORMAL\nContent C", metadata={"page": 3}),
        Document(page_content="NORMAL\nContent D", metadata={"page": 4}),
    ]
    
    cleaned = cleaner.clean(pages)
    
    # RARE HEADER occurs 50% (2/4) - should be kept
    assert "RARE HEADER" in cleaned
    # NORMAL occurs 50% (2/4) - should be kept
    assert "NORMAL" in cleaned


def test_cleaner_short_document_no_header_removal():
    """Test that documents with < 3 pages bypass the header removal heuristic."""
    cleaner = TextCleaner()
    pages = [
        Document(page_content="HEADER\nContent 1", metadata={"page": 1}),
        Document(page_content="HEADER\nContent 2", metadata={"page": 2}),
    ]
    
    cleaned = cleaner.clean(pages)
    
    # Even though HEADER is on 100% of pages, it's < 3 pages, so it should stay.
    assert "HEADER" in cleaned
