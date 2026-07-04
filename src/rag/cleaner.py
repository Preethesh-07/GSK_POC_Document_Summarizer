"""
Text cleaning and normalization.

Deterministic processing — no LLM involved.
Handles: whitespace normalization, header/footer removal,
non-printable character stripping.
"""

import re

from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """Normalizes and cleans extracted PDF text."""

    # Minimum percentage of pages that must share a line
    # for it to be considered a repeated header/footer
    REPETITION_THRESHOLD = 0.7

    # Number of characters to check at start/end of each page
    HEADER_FOOTER_CHARS = 100

    def clean(self, pages: list[Document]) -> str:
        """Clean and normalize text from extracted PDF pages.

        Args:
            pages: List of Document objects from the PDF loader.

        Returns:
            Cleaned, normalized text as a single string.
        """
        if not pages:
            logger.warning("No pages to clean")
            return ""

        # ── Step 1: Remove repeated headers/footers ───────────────────
        cleaned_pages = self._remove_repeated_headers_footers(pages)

        # ── Step 2: Join all page texts ───────────────────────────────
        full_text = "\n\n".join(
            page.page_content for page in cleaned_pages
        )

        # ── Step 3: Normalize whitespace ──────────────────────────────
        full_text = self._normalize_whitespace(full_text)

        # ── Step 4: Strip non-printable characters ────────────────────
        full_text = self._strip_non_printable(full_text)

        logger.info(
            f"Text cleaned: {len(full_text)} chars "
            f"from {len(pages)} pages"
        )
        return full_text

    def _remove_repeated_headers_footers(
        self, pages: list[Document]
    ) -> list[Document]:
        """Remove lines that appear at the top/bottom of most pages.

        Heuristic: if the first or last line of a page appears in more
        than REPETITION_THRESHOLD of all pages, it's likely a
        header/footer and is removed.
        """
        if len(pages) < 3:
            return pages

        # Collect first and last lines from each page
        first_lines: list[str] = []
        last_lines: list[str] = []

        for page in pages:
            lines = page.page_content.strip().split("\n")
            if lines:
                first_lines.append(lines[0].strip())
                last_lines.append(lines[-1].strip())

        # Find repeated headers
        repeated_headers = self._find_repeated_lines(
            first_lines, len(pages)
        )

        # Find repeated footers
        repeated_footers = self._find_repeated_lines(
            last_lines, len(pages)
        )

        if repeated_headers:
            logger.info(
                f"Removing {len(repeated_headers)} repeated header(s)"
            )
        if repeated_footers:
            logger.info(
                f"Removing {len(repeated_footers)} repeated footer(s)"
            )

        # Remove repeated lines from pages
        cleaned: list[Document] = []
        for page in pages:
            lines = page.page_content.strip().split("\n")
            filtered_lines = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                if i == 0 and stripped in repeated_headers:
                    continue
                if i == len(lines) - 1 and stripped in repeated_footers:
                    continue
                filtered_lines.append(line)

            cleaned.append(
                Document(
                    page_content="\n".join(filtered_lines),
                    metadata=page.metadata,
                )
            )

        return cleaned

    def _find_repeated_lines(
        self, lines: list[str], total_pages: int
    ) -> set[str]:
        """Find lines that appear in more than threshold% of pages."""
        from collections import Counter

        counter = Counter(lines)
        threshold = total_pages * self.REPETITION_THRESHOLD
        return {
            line
            for line, count in counter.items()
            if count >= threshold and line  # skip empty lines
        }

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace characters."""
        # Collapse multiple newlines into double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces into single space
        text = re.sub(r"[^\S\n]+", " ", text)
        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines)

    def _strip_non_printable(self, text: str) -> str:
        """Remove non-printable characters, keeping standard whitespace."""
        # Keep printable ASCII + common unicode letters + whitespace
        return re.sub(r"[^\x20-\x7E\n\t\u00A0-\uFFFF]", "", text)
