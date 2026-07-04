"""
Text-level validation after extraction.

Checks: minimum length, non-gibberish content, and basic quality.
"""

import re
from dataclasses import dataclass

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextValidationResult:
    """Result of text content validation."""

    is_valid: bool
    error: str | None = None


class TextValidator:
    """Validates extracted text content quality."""

    # Minimum ratio of alphabetic characters to total characters
    MIN_ALPHA_RATIO = 0.5

    def validate(self, text: str) -> TextValidationResult:
        """Run all text-level validations.

        Args:
            text: Cleaned/extracted text from the document.

        Returns:
            TextValidationResult with is_valid flag and any error.
        """
        # ── Check minimum length ──────────────────────────────────────
        if len(text.strip()) < settings.min_text_length:
            logger.error(
                f"Text too short: {len(text.strip())} chars "
                f"(min: {settings.min_text_length})"
            )
            return TextValidationResult(
                is_valid=False,
                error=(
                    f"Extracted text is too short ({len(text.strip())} chars). "
                    f"Minimum required: {settings.min_text_length}"
                ),
            )

        # ── Check for gibberish (low alphabetic ratio) ────────────────
        alpha_chars = sum(1 for c in text if c.isalpha())
        total_chars = len(text.strip())

        if total_chars > 0:
            alpha_ratio = alpha_chars / total_chars
            if alpha_ratio < self.MIN_ALPHA_RATIO:
                logger.error(
                    f"Text appears to be gibberish: "
                    f"alpha ratio {alpha_ratio:.2f} "
                    f"(min: {self.MIN_ALPHA_RATIO})"
                )
                return TextValidationResult(
                    is_valid=False,
                    error=(
                        f"Text quality too low — alphabetic ratio is "
                        f"{alpha_ratio:.2f} (minimum: {self.MIN_ALPHA_RATIO}). "
                        f"Document may be scanned or image-based."
                    ),
                )

        # ── Check for reasonable word count ───────────────────────────
        words = re.findall(r"\b\w+\b", text)
        if len(words) < 20:
            logger.error(f"Too few words extracted: {len(words)}")
            return TextValidationResult(
                is_valid=False,
                error=f"Only {len(words)} words extracted — insufficient for summarization.",
            )

        logger.info(
            f"Text validated: {len(text)} chars, "
            f"{len(words)} words, "
            f"alpha ratio {alpha_ratio:.2f}"
        )
        return TextValidationResult(is_valid=True)
