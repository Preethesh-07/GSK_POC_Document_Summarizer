"""
Model Benchmarking utility.

Runs the same document through different LLMs to compare
summarization quality, latency, and token usage.
"""

import time
from pathlib import Path

from src.graph.workflow import compile_workflow
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Benchmarker:
    """Utility to benchmark different LLMs for summarization."""

    # Configured from EDS + user choices
    MODELS = [
        "openai/gpt-oss-120b",        # Primary
        "llama-3.3-70b-versatile",    # Benchmark comparison
    ]

    async def benchmark(self, pdf_path: str, reference_summary: str | None = None) -> dict:
        """Run a document through all benchmark models.

        Args:
            pdf_path: Path to the test PDF document.
            reference_summary: Optional ground truth for automatic evaluation.

        Returns:
            Dictionary mapping model names to their performance metrics.
        """
        import os
        from src.evaluation.rouge import RougeEvaluator
        
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot benchmark, file missing: {pdf_path}")

        logger.info(f"Starting benchmark on {path.name} with models: {self.MODELS}")
        app = compile_workflow()
        
        results = {}
        rouge = RougeEvaluator() if reference_summary else None

        for model in self.MODELS:
            logger.info(f"Benchmarking model: {model}")
            
            # Temporarily override the environment variable for this run
            # Note: A cleaner way in production would be to pass the model 
            # down through the GraphState or via a configurable graph config.
            original_model = os.environ.get("LLM_MODEL")
            os.environ["LLM_MODEL"] = model
            
            initial_state = {
                "document_id": f"benchmark_{model.replace('/', '_')}_{path.stem}",
                "filename": path.name,
                "pdf_path": str(path.absolute()),
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

            start_time = time.time()
            try:
                state_result = await app.ainvoke(initial_state)
                latency = time.time() - start_time
                
                final_summary = state_result.get("final_summary", "")
                status = state_result.get("processing_status", "unknown")
                
                metrics = {
                    "status": status,
                    "latency_seconds": latency,
                    "summary_length": len(final_summary),
                }
                
                # Automatic evaluation if a reference is provided
                if rouge and final_summary and status == "completed":
                    rouge_scores = rouge.evaluate(reference_summary, final_summary)
                    metrics["rouge"] = rouge_scores
                    
                results[model] = metrics
                logger.info(f"Model {model} finished in {latency:.2f}s (status: {status})")
                
            except Exception as e:
                logger.error(f"Benchmark failed for model {model}: {e}")
                results[model] = {"status": "failed", "error": str(e)}
            finally:
                if original_model is not None:
                    os.environ["LLM_MODEL"] = original_model
                else:
                    os.environ.pop("LLM_MODEL", None)

        return results
