"""
LLM Comparative Evaluation Script.

Compares summarization quality between different LLMs using LangSmith's
evaluate() and experiment comparison features.

Usage:
    uv run python -m src.evaluation.evaluate_llms
"""

import asyncio

from langsmith import Client, evaluate

from src.rag.loader import PDFLoader
from src.llm.groq_client import get_llm
from src.agents.summarization import ChunkSummarizer
from src.config.settings import settings


def ensure_dataset(client: Client, pdf_path: str) -> str:
    """Ensure the LangSmith dataset exists, or create it from the PDF.

    Parses the PDF once, extracts chunks, and uploads them as dataset
    examples so we never have to re-parse the PDF during evaluation.

    Args:
        client: Authenticated LangSmith client.
        pdf_path: Path to the source PDF.

    Returns:
        The dataset name in LangSmith.
    """
    dataset_name = "Pharma Summarization Full Benchmark"

    if not client.has_dataset(dataset_name=dataset_name):
        print(f"Creating dataset '{dataset_name}' from {pdf_path}...")
        dataset = client.create_dataset(dataset_name=dataset_name)

        # Load and chunk the PDF using our UnstructuredLoader
        loader = PDFLoader()
        chunks = loader.load(pdf_path)

        # Use ALL chunks for comprehensive comparison
        examples = []
        for i, chunk in enumerate(chunks):
            examples.append({
                "inputs": {"text": chunk.page_content},
            })

        client.create_examples(dataset_id=dataset.id, examples=examples)
        print(f"Uploaded {len(examples)} chunk examples to dataset '{dataset_name}'")
    else:
        print(f"Dataset '{dataset_name}' already exists.")

    return dataset_name


async def _summarize_with_model(inputs: dict, model_name: str) -> str:
    """Helper to run the chunk summarizer with a specific model."""
    llm = get_llm(model=model_name)
    summarizer = ChunkSummarizer(llm)
    chunk_data = {"text": inputs["text"], "id": "eval_chunk"}
    return await summarizer.summarize_chunk(chunk_data)


# LangSmith evaluate() expects synchronous functions, so we wrap the async calls
def run_llama(inputs: dict):
    """Run summarization with Llama 3.3 70B."""
    result = asyncio.run(_summarize_with_model(inputs, "llama-3.3-70b-versatile"))
    return {"summary": result}


def run_gpt_oss(inputs: dict):
    """Run summarization with OpenAI GPT-OSS 120B."""
    result = asyncio.run(_summarize_with_model(inputs, "openai/gpt-oss-120b"))
    return {"summary": result}


if __name__ == "__main__":
    pdf_path = "pharma research paper.pdf"

    # Create a single authenticated LangSmith client
    ls_client = Client(api_key=settings.langsmith_api_key)

    # 1. Setup the dataset from the PDF (only runs once)
    dataset_name = ensure_dataset(ls_client, pdf_path)

    # 2. Run evaluation for Llama 3.3 70B
    print("\n--- Evaluating Llama 3.3 70B Versatile ---")
    evaluate(
        run_llama,
        data=dataset_name,
        client=ls_client,
        experiment_prefix="Llama-3.3-70b",
    )

    # 3. Run evaluation for GPT-OSS 120B
    print("\n--- Evaluating OpenAI/GPT-OSS 120B ---")
    evaluate(
        run_gpt_oss,
        data=dataset_name,
        client=ls_client,
        experiment_prefix="GPT-OSS-120b",
    )

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Go to your LangSmith Dashboard -> Datasets & Testing")
    print("2. Click on 'Pharma Summarization Benchmark'")
    print("3. Go to the 'Experiments' tab")
    print("4. Select both 'Llama-3.3-70b' and 'GPT-OSS-120b' experiments")
    print("5. Click 'Compare' to see the side-by-side results!")
