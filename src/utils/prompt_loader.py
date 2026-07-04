"""
Utility for loading prompt templates from the prompts directory.
"""

from pathlib import Path

# Root of the src/prompts directory
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(prompt_name: str) -> str:
    """Load a text-based prompt template by filename.

    Args:
        prompt_name: Name of the prompt file (e.g., 'chunk_summary.txt')

    Returns:
        The content of the prompt file as a string.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    prompt_path = PROMPTS_DIR / prompt_name
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    return prompt_path.read_text(encoding="utf-8")
