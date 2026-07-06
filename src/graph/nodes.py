"""
LangGraph node functions.

Each function represents a node in the graph, taking the current
GraphState and returning a dictionary of state updates.
"""

from src.agents.review import ReviewAgent
from src.agents.summarization import ChunkSummarizer, HierarchicalMerger
from src.guardrails.manager import GuardrailManager
from src.llm.groq_client import get_llm
from src.rag.loader import PDFLoader
from src.rag.vectorstore import VectorStoreManager
from src.utils.logger import get_logger

from src.graph.state import GraphState

logger = get_logger(__name__)


async def input_guardrails(state: GraphState) -> dict:
    """Node 1: Validate the PDF document."""
    logger.info("Executing node: input_guardrails")
    manager = GuardrailManager()
    
    result = manager.validate(state["pdf_path"])
    
    if not result.is_valid:
        return {
            "processing_status": "failed",
            "errors": result.errors,
            "metadata": result.metadata,
        }
        
    return {
        "processing_status": "validated",
        "metadata": result.metadata,
        # Pass the extracted text forward to save redundant loading
        "cleaned_text": result.extracted_text, 
    }


async def ingest_document(state: GraphState) -> dict:
    """Node 2: Load PDF and extract semantic chunks using Unstructured."""
    logger.info("Executing node: ingest_document")
    
    loader = PDFLoader()
    # Loader now returns pre-chunked semantic documents
    semantic_chunks = loader.load(state["pdf_path"])
    
    # We bypass the old TextCleaner because Unstructured handles headers/footers
    return {
        "raw_pages": semantic_chunks,  # Storing the semantic chunks here temporarily
        "processing_status": "processing",
    }


async def chunk_document(state: GraphState) -> dict:
    """Node 3: Format semantic chunks into internal schema."""
    logger.info("Executing node: chunk_document")
    
    semantic_docs = state.get("raw_pages", [])
    if not semantic_docs:
        return {
            "processing_status": "failed",
            "errors": ["Failed to extract semantic chunks from the document"],
        }

    chunks = []
    for i, doc in enumerate(semantic_docs):
        # Unstructured provides rich metadata, including page numbers
        chunk = {
            "id": f"{state['document_id']}_chunk_{i}",
            "index": i,
            "page": doc.metadata.get("page_number", -1),
            "token_estimate": len(doc.page_content) // 4,
            "text": doc.page_content,
        }
        chunks.append(chunk)
        
    return {"chunks": chunks}


async def generate_embeddings(state: GraphState) -> dict:
    """Node 4A: Generate and store embeddings in pgvector."""
    logger.info("Executing node: generate_embeddings")
    
    try:
        vectorstore = VectorStoreManager()
        vectorstore.store_chunks(state["chunks"], state["document_id"])
        return {"embeddings_stored": True}
    except Exception as e:
        logger.error(f"Failed to generate/store embeddings: {e}")
        return {
            "embeddings_stored": False,
            "errors": [f"Embedding failure: {str(e)}"],
        }


async def summarize_chunks(state: GraphState) -> dict:
    """Node 4B: Summarize each chunk in parallel."""
    logger.info("Executing node: summarize_chunks")
    
    try:
        llm = get_llm()
        summarizer = ChunkSummarizer(llm)
        summaries = await summarizer.summarize_all(state["chunks"])
        return {"chunk_summaries": summaries}
    except Exception as e:
        logger.error(f"Failed to summarize chunks: {e}")
        return {
            "processing_status": "failed",
            "errors": [f"Chunk summarization failure: {str(e)}"],
        }


async def hierarchical_merge(state: GraphState) -> dict:
    """Node 5: Hierarchically merge chunk summaries."""
    logger.info("Executing node: hierarchical_merge")
    
    # If a parallel node failed, don't proceed
    if state.get("processing_status") == "failed":
        return {}

    try:
        llm = get_llm(model="openai/gpt-oss-120b")
        merger = HierarchicalMerger(llm)
        merged = await merger.merge(state["chunk_summaries"])
        return {"merged_summary": merged}
    except Exception as e:
        logger.error(f"Failed to merge summaries: {e}")
        return {
            "processing_status": "failed",
            "errors": [f"Hierarchical merge failure: {str(e)}"],
        }


async def generate_executive_summary(state: GraphState) -> dict:
    """Node 6: Generate final executive summary (handled by merge node in this architecture)."""
    logger.info("Executing node: generate_executive_summary")
    
    # In this design, the hierarchical merge actually produces the 
    # executive summary in its final step. We just pass it through here,
    # or this node could apply formatting. For now, it's a pass-through.
    
    if state.get("processing_status") == "failed":
        return {}
        
    return {"final_summary": state["merged_summary"]}


async def review_summary(state: GraphState) -> dict:
    """Node 7: Review and polish the summary."""
    logger.info("Executing node: review_summary")
    
    if state.get("processing_status") == "failed":
        return {}

    try:
        llm = get_llm(model="openai/gpt-oss-120b")
        reviewer = ReviewAgent(llm)
        polished = await reviewer.review(state["final_summary"])
        
        return {
            "final_summary": polished,
            "processing_status": "completed",
        }
    except Exception as e:
        logger.error(f"Failed to review summary: {e}")
        return {
            "processing_status": "failed",
            "errors": [f"Review failure: {str(e)}"],
        }
