"""
Unit tests for the PDFLoader.
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.rag.loader import PDFLoader


@patch("src.rag.loader.PyPDFLoader")
def test_loader_loads_successfully(mock_pypdf_loader):
    """Test that PDFLoader successfully loads and returns pages."""
    # Arrange: Mock the behavior of PyPDFLoader
    mock_instance = mock_pypdf_loader.return_value
    mock_documents = [
        Document(page_content="Page 1 content", metadata={"page": 0}),
        Document(page_content="Page 2 content", metadata={"page": 1}),
    ]
    mock_instance.load.return_value = mock_documents
    
    loader = PDFLoader()
    
    # Act
    pages = loader.load("dummy.pdf")
    
    # Assert
    mock_pypdf_loader.assert_called_once_with("dummy.pdf")
    mock_instance.load.assert_called_once()
    assert len(pages) == 2
    assert pages[0].page_content == "Page 1 content"
    assert pages[1].page_content == "Page 2 content"


@patch("src.rag.loader.PyPDFLoader")
def test_loader_raises_on_load_error(mock_pypdf_loader):
    """Test that the loader propagates exceptions from PyPDFLoader."""
    # Arrange
    mock_instance = mock_pypdf_loader.return_value
    mock_instance.load.side_effect = Exception("PDF parsing error")
    
    loader = PDFLoader()
    
    # Act & Assert
    with pytest.raises(Exception, match="PDF parsing error"):
        loader.load("corrupt_file.pdf")
        
    mock_pypdf_loader.assert_called_once_with("corrupt_file.pdf")


def test_loader_raises_on_missing_file_integration():
    """Test integration without mocks to ensure real unhandled missing files throw exceptions."""
    loader = PDFLoader()
    # PyPDFLoader usually raises ValueError or FileNotFoundError if the file doesn't exist.
    # Catching Exception guarantees we cover whatever the underlying lib throws.
    with pytest.raises(Exception):
        loader.load("this_file_surely_does_not_exist_999.pdf")

