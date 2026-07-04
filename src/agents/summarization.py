"""
Summarization agents.

Includes:
1. ChunkSummarizer: Summarizes individual text chunks in parallel.
2. HierarchicalMerger: Batches and merges multiple summaries iteratively.
"""

import asyncio

from langchain_groq import ChatGroq

from src.config.settings import settings
from src.utils.logger import get_logger
from src.utils.prompt_loader import load_prompt

logger = get_logger(__name__)


class ChunkSummarizer:
    """Agent that summarizes individual document chunks."""

    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.prompt_template = load_prompt("chunk_summary.txt")

    async def summarize_chunk(self, chunk: dict) -> str:
        """Summarize a single chunk.

        Args:
            chunk: Chunk dictionary containing 'text' and 'id'.

        Returns:
            Concise summary of the chunk.
        """
        logger.debug(f"Summarizing chunk: {chunk['id']}")
        prompt = self.prompt_template.format(chunk_text=chunk["text"])
        response = await self.llm.ainvoke(prompt)
        return str(response.content)

    async def summarize_all(self, chunks: list[dict]) -> list[str]:
        """Summarize multiple chunks in parallel.

        Args:
            chunks: List of chunk dictionaries.

        Returns:
            List of chunk summaries in the same order.
        """
        if not chunks:
            return []

        logger.info(f"Summarizing {len(chunks)} chunks in parallel...")
        # Run all chunk summaries concurrently using asyncio.gather
        tasks = [self.summarize_chunk(c) for c in chunks]
        summaries = await asyncio.gather(*tasks)
        logger.info(f"Generated {len(summaries)} chunk summaries")
        return list(summaries)


class HierarchicalMerger:
    """Agent that hierarchically merges multiple summaries."""

    def __init__(self, llm: ChatGroq, batch_size: int | None = None):
        self.llm = llm
        self.batch_size = batch_size or settings.merge_batch_size
        self.prompt_template = load_prompt("merge_summary.txt")

    async def merge(self, summaries: list[str]) -> str:
        """Merge multiple summaries iteratively into a single executive summary.

        Args:
            summaries: List of chunk summaries.

        Returns:
            Final merged executive summary.
        """
        if not summaries:
            return ""

        current_summaries = summaries
        iteration = 1

        while len(current_summaries) > 1:
            logger.info(
                f"Hierarchical merge round {iteration}: "
                f"merging {len(current_summaries)} summaries "
                f"(batch size {self.batch_size})"
            )
            batches = self._create_batches(current_summaries)
            merged = []
            
            # Process batches sequentially or in parallel?
            # Processing sequentially to avoid hitting rate limits on large docs
            # but could be parallelized. Let's do sequential for safety.
            for i, batch in enumerate(batches):
                logger.debug(
                    f"Processing merge batch {i+1}/{len(batches)} "
                    f"({len(batch)} items)"
                )
                joined_summaries = "\n\n---\n\n".join(batch)
                prompt = self.prompt_template.format(
                    summaries=joined_summaries,
                    max_tokens=settings.max_summary_tokens,
                )
                response = await self.llm.ainvoke(prompt)
                merged.append(str(response.content))
            
            current_summaries = merged
            iteration += 1

        logger.info("Hierarchical merge complete")
        return current_summaries[0]

    def _create_batches(self, items: list[str]) -> list[list[str]]:
        """Group items into batches of self.batch_size."""
        return [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
