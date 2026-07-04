"""
Review agent.

Reviews and polishes the final executive summary for quality.
Ensures readability, coherence, and lack of repetition without
introducing new facts.
"""

from langchain_groq import ChatGroq

from src.utils.logger import get_logger
from src.utils.prompt_loader import load_prompt

logger = get_logger(__name__)


class ReviewAgent:
    """Agent that reviews and polishes the final summary."""

    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.prompt_template = load_prompt("review_summary.txt")

    async def review(self, summary: str) -> str:
        """Review and improve the executive summary.

        Args:
            summary: The final merged executive summary.

        Returns:
            The polished, reviewed summary.
        """
        logger.info("Starting review of executive summary...")
        
        if not summary.strip():
            logger.warning("Empty summary provided for review")
            return ""

        prompt = self.prompt_template.format(summary=summary)
        response = await self.llm.ainvoke(prompt)
        
        reviewed_summary = str(response.content)
        logger.info("Executive summary review complete")
        return reviewed_summary
