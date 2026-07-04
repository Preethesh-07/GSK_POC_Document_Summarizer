"""
CLI Entry Point.

Runs the AI Document Summarization System on a given PDF file.
"""

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from src.graph.workflow import compile_workflow
from src.utils.logger import get_logger

logger = get_logger(__name__)
console = Console()


async def run(pdf_path: str) -> str:
    """Run the summarization pipeline on a PDF file.

    Args:
        pdf_path: Path to the PDF to summarize.

    Returns:
        The final executive summary.
    """
    path = Path(pdf_path)
    
    # ── Initialize State ──────────────────────────────────────────────
    initial_state = {
        "document_id": path.stem,
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

    # ── Compile and Run ───────────────────────────────────────────────
    console.print(f"[bold blue]Starting summarization pipeline for:[/bold blue] {path.name}")
    app = compile_workflow()
    
    try:
        # LangGraph invoke returns the final state
        result = await app.ainvoke(initial_state)
    except Exception as e:
        logger.exception("Graph execution failed with an unhandled exception.")
        console.print(Panel(f"[red]FATAL ERROR[/red]\n{str(e)}", title="Execution Failed"))
        return ""

    # ── Handle Results ────────────────────────────────────────────────
    status = result.get("processing_status")
    
    if status == "failed":
        errors = "\n".join(f"- {err}" for err in result.get("errors", []))
        console.print(
            Panel(
                f"[red]Pipeline failed during execution.[/red]\n\nErrors:\n{errors}",
                title="Validation / Processing Error",
                border_style="red",
            )
        )
        return ""

    final_summary = result.get("final_summary", "")
    
    if final_summary:
        console.print(
            Panel(
                final_summary,
                title="[bold green]Executive Summary[/bold green]",
                border_style="green",
            )
        )
        
    return final_summary


def main():
    """Main CLI entry function."""
    if len(sys.argv) < 2:
        console.print("[red]Usage: uv run python -m src.main <path_to_pdf>[/red]")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        console.print(f"[red]Error: File not found - {pdf_path}[/red]")
        sys.exit(1)
        
    try:
        asyncio.run(run(pdf_path))
    except KeyboardInterrupt:
        console.print("\n[yellow]Execution cancelled by user.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
