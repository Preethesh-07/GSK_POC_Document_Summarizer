# Production-Grade AI Document Summarization System

## Engineering Design Specification (EDS)

Version: 1.0

> This document is the implementation specification for a
> production-grade document summarization system. It is intended to be
> used by AI coding agents (Cursor, Antigravity, Claude Code, etc.) as
> the source of truth.

# 1. Objective

Build a modular, production-grade summarization platform for long PDF
documents.

Primary focus: - High-quality summarization - Clean architecture -
Extensibility - Observability - Maintainability

Future capabilities: - Document Q&A - OCR - Output guardrails -
Multi-document summarization

------------------------------------------------------------------------

# 2. Functional Requirements

## Input

-   PDF documents
-   Digital PDFs only (OCR later)

## Output

-   Executive summary
-   Maximum one page
-   Contextually accurate
-   Concise
-   Professional tone

------------------------------------------------------------------------

# 3. Non-Functional Requirements

-   Modular architecture
-   Deterministic processing where possible
-   Configurable parameters
-   Retry mechanisms
-   Logging
-   Traceability
-   Testability

------------------------------------------------------------------------

# 4. Technology Stack

Backend: - Python - FastAPI (optional) - CLI (Phase 1)

LLM: - Groq - GPT-OSS (preferred) - Llama (benchmark)

Embeddings: - Gemini Embeddings

Vector Database: - Neon PostgreSQL - pgvector

Framework: - LangChain - LangGraph

Observability: - LangSmith

------------------------------------------------------------------------

# 5. High-Level Pipeline

Upload PDF → Input Guardrails → PDF Loader → Text Cleaning →
RecursiveCharacterTextSplitter → Parallel: A. Embedding Pipeline B.
Summarization Pipeline → Hierarchical Merge → Executive Summary → Review
Agent → Final Summary

------------------------------------------------------------------------

# 6. Repository Layout

src/ agents/ ingestion.py summarization.py review.py

    graph/
        state.py
        workflow.py
        nodes.py

    guardrails/
        manager.py
        pdf_validator.py
        text_validator.py
        prompt_injection.py

    rag/
        loader.py
        cleaner.py
        chunker.py
        embeddings.py
        vectorstore.py

    prompts/
        chunk_summary.txt
        merge_summary.txt
        review_summary.txt

    llm/
        groq_client.py

    evaluation/
        benchmark.py
        rouge.py
        bertscore.py

    config/
        settings.py

    utils/

    main.py

------------------------------------------------------------------------

# 7. GraphState

Fields

-   document_id
-   filename
-   raw_pages
-   cleaned_text
-   chunks
-   embeddings
-   chunk_summaries
-   merged_summary
-   final_summary
-   metadata
-   processing_status
-   errors

Every node reads/writes only the fields it owns.

------------------------------------------------------------------------

# 8. Node Specifications

## Node 1 --- Input Guardrails

Purpose: Validate the document before processing.

Checks: - valid PDF - readable - size - page count - extracted text
exists - prompt injection patterns

Output: Validated document.

Failure: Return descriptive error.

------------------------------------------------------------------------

## Node 2 --- Ingestion

Responsibilities: - load PDF - normalize text - remove repeated
whitespace - remove repeated headers/footers where feasible

No LLM.

------------------------------------------------------------------------

## Node 3 --- Chunking

Use RecursiveCharacterTextSplitter.

Configuration: - chunk_size - overlap

Chunk metadata: - id - index - page - token estimate - text

------------------------------------------------------------------------

## Node 4A --- Embedding

Generate Gemini embeddings.

Persist to pgvector.

Metadata: - document_id - chunk_id - page - embedding

------------------------------------------------------------------------

## Node 4B --- Chunk Summarization

Input: Single chunk.

Output: Concise chunk summary.

Rules: - preserve meaning - avoid hallucination - concise

Runs in parallel.

------------------------------------------------------------------------

## Node 5 --- Hierarchical Merge

Algorithm:

1.  Take N chunk summaries.
2.  Group by configurable batch size.
3.  Merge each batch.
4.  Repeat until one summary remains.

Never merge all summaries in one request.

------------------------------------------------------------------------

## Node 6 --- Executive Summary

Generate: - one-page summary - logical flow - concise - complete

------------------------------------------------------------------------

## Node 7 --- Review

Responsibilities: - readability - coherence - repetition - factual
consistency - length

No new information may be introduced.

------------------------------------------------------------------------

# 9. Hierarchical Summarization Algorithm

Pseudo:

while summaries \> 1: group summaries summarize each group replace old
summaries

return final summary

Benefits: - scalable - bounded context - predictable latency

------------------------------------------------------------------------

# 10. Prompt Library

chunk_summary.txt

Purpose: Summarize a chunk.

merge_summary.txt

Purpose: Merge summaries without repetition.

review_summary.txt

Purpose: Improve quality only.

Prompts must be version controlled.

------------------------------------------------------------------------

# 11. RAG Pipeline

Purpose: Future retrieval.

Flow:

Chunks → Gemini Embeddings → pgvector

No retrieval during summarization.

------------------------------------------------------------------------

# 12. Database

documents

-   id
-   filename
-   uploaded_at

chunks

-   id
-   document_id
-   chunk_index
-   page_number
-   chunk_text
-   embedding

------------------------------------------------------------------------

# 13. Configuration

Everything configurable:

-   model
-   embedding model
-   chunk size
-   overlap
-   retries
-   max pages
-   summary length

Never hardcode.

------------------------------------------------------------------------

# 14. Error Handling

Retry: - embedding API - LLM - DB

Reject: - invalid PDF - empty document

Log every failure.

------------------------------------------------------------------------

# 15. Observability

LangSmith:

Capture: - prompts - latency - tokens - node execution - failures

------------------------------------------------------------------------

# 16. Evaluation

Benchmark: - GPT-OSS - Llama

Automatic: - ROUGE - BERTScore

Human: - coherence - factuality - completeness - readability

------------------------------------------------------------------------

# 17. Testing

Unit: - loader - cleaner - chunker - guardrails

Integration: - end-to-end summarization

Regression: - same document should produce stable quality.

------------------------------------------------------------------------

# 18. Build Order

1.  Configuration
2.  Input Guardrails
3.  Loader
4.  Cleaner
5.  Chunker
6.  Embeddings
7.  pgvector
8.  Chunk Summaries
9.  Hierarchical Merge
10. Executive Summary
11. Review
12. LangGraph
13. LangSmith
14. Benchmark
15. Tests

------------------------------------------------------------------------

# 19. Future Roadmap

Phase 2 - Output Guardrails - OCR - Q&A

Phase 3 - Multi-document summarization - Hybrid Retrieval - Feedback
loop

------------------------------------------------------------------------

# 20. Success Criteria

The implementation is considered complete when:

-   Produces accurate summaries for 25--60 page PDFs.
-   Summary is \<= one page.
-   Graph execution is observable.
-   Embeddings stored successfully.
-   Models benchmarked and justified.
-   Architecture is modular and extensible.
