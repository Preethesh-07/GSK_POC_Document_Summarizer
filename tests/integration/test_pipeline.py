"""
Integration tests for the full pipeline.

These tests require real API keys to run successfully.
They are marked with a custom pytest mark (e.g., @pytest.mark.integration).
"""

import os
from pathlib import Path

import pytest
from langgraph.graph.state import CompiledStateGraph

from src.graph.workflow import compile_workflow


@pytest.fixture
def workflow() -> CompiledStateGraph:
    return compile_workflow()


@pytest.mark.asyncio
async def test_full_pipeline_compiles_and_aborts_on_missing_file(workflow):
    """Test that the workflow compiles and correctly aborts early on invalid input."""
    
    initial_state = {
        "document_id": "test_doc",
        "filename": "missing.pdf",
        "pdf_path": "this_file_does_not_exist.pdf",
        "raw_pages": [],
        "cleaned_text": "",
        "chunks": [],
        "embeddings_stored": False,
        "chunk_summaries": [],
        "merged_summary": "",
        "final_summary": "",
        "metadata": {},
        "processing_status": "pending",
        "errors": [],
    }

    result = await workflow.ainvoke(initial_state)
    
    assert result["processing_status"] == "failed"
    assert len(result["errors"]) > 0
    assert "File not found" in result["errors"][0]
    
    # Since it failed at guardrails, it should not have chunks or summaries
    assert len(result["chunks"]) == 0
    assert result["final_summary"] == ""
