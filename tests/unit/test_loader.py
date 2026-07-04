"""
Unit tests for the PDFLoader.
"""

import pytest
from langchain_core.documents import Document

from src.rag.loader import PDFLoader


def test_loader_raises_on_missing_file():
    """Test that the loader raises an exception for a missing file."""
    loader = PDFLoader()
    with pytest.raises(Exception):
        loader.load("non_existent_file_12345.pdf")


# If a dummy PDF is added to data/sample_pdfs/, you could add:
# def test_loader_loads_valid_pdf():
#     loader = PDFLoader()
#     pages = loader.load("data/sample_pdfs/dummy.pdf")
#     assert len(pages) > 0
#     assert isinstance(pages[0], Document)
