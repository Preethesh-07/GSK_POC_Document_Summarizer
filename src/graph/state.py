"""
LangGraph state definition.

Defines the structure of the data passed between nodes in the graph.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """The state dictionary for the document summarization workflow."""

    # ── Identity & Input ──────────────────────────────────────────────
    document_id: str
    filename: str
    pdf_path: str

    # ── Processing Data ───────────────────────────────────────────────
    raw_pages: list[Document]
    cleaned_text: str
    chunks: list[dict]
    
    # ── Outputs ───────────────────────────────────────────────────────
    embeddings_stored: bool
    chunk_summaries: list[str]
    merged_summary: str
    final_summary: str
    
    # ── Metadata & Status ─────────────────────────────────────────────
    metadata: dict
    
    # Status can be: "pending", "processing", "validated", "completed", "failed"
    processing_status: str
    
    # Errors are accumulated using operator.add (list concatenation)
    errors: Annotated[list[str], operator.add]
