"""
LangGraph workflow builder.

Wires up the StateGraph with all nodes, conditional edges,
and parallel execution paths.
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.graph.nodes import (
    chunk_document,
    generate_embeddings,
    generate_executive_summary,
    hierarchical_merge,
    ingest_document,
    input_guardrails,
    review_summary,
    summarize_chunks,
)
from src.graph.state import GraphState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_workflow() -> StateGraph:
    """Construct the summarization workflow graph.

    Returns:
        Uncompiled StateGraph instance.
    """
    workflow = StateGraph(state_schema=GraphState)

    # ── Add Nodes ─────────────────────────────────────────────────────
    workflow.add_node("input_guardrails", input_guardrails)
    workflow.add_node("ingest_document", ingest_document)
    workflow.add_node("chunk_document", chunk_document)
    workflow.add_node("generate_embeddings", generate_embeddings)
    workflow.add_node("summarize_chunks", summarize_chunks)
    workflow.add_node("hierarchical_merge", hierarchical_merge)
    workflow.add_node("executive_summary", generate_executive_summary)
    workflow.add_node("review_summary", review_summary)

    # ── Entry Point ───────────────────────────────────────────────────
    workflow.set_entry_point("input_guardrails")

    # ── Edges ─────────────────────────────────────────────────────────
    # Conditional logic based on guardrail validation
    def check_guardrails(state: GraphState) -> str:
        status = state.get("processing_status")
        if status == "validated":
            return "continue"
        return "abort"

    workflow.add_conditional_edges(
        "input_guardrails",
        check_guardrails,
        {
            "continue": "ingest_document",
            "abort": END,
        },
    )

    # Linear flow for ingestion and chunking
    workflow.add_edge("ingest_document", "chunk_document")

    # Parallel fan-out: chunking leads to both embedding and summarization
    # Note: LangGraph automatically handles parallel execution when multiple
    # edges originate from the same node in async execution.
    workflow.add_edge("chunk_document", "generate_embeddings")
    workflow.add_edge("chunk_document", "summarize_chunks")

    # Parallel fan-in: both branches converge at hierarchical merge
    workflow.add_edge("generate_embeddings", "hierarchical_merge")
    workflow.add_edge("summarize_chunks", "hierarchical_merge")

    # Linear flow to completion
    workflow.add_edge("hierarchical_merge", "executive_summary")
    workflow.add_edge("executive_summary", "review_summary")
    workflow.add_edge("review_summary", END)

    logger.debug("StateGraph workflow built successfully")
    return workflow


def compile_workflow() -> CompiledStateGraph:
    """Build and compile the workflow.

    Returns:
        CompiledStateGraph ready for invocation.
    """
    workflow = build_workflow()
    compiled = workflow.compile()
    logger.info("StateGraph workflow compiled successfully")
    return compiled
