"""
PDF file-level validation.

Checks: file existence, valid PDF format, readability,
file size limits, and page count limits.
"""

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    error: str | None = None
    metadata: dict = field(default_factory=dict)


class PDFValidator:
    """Validates a PDF file before any processing begins."""

    def validate(self, pdf_path: str) -> ValidationResult:
        """Run all PDF-level validations.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ValidationResult with is_valid flag and any error details.
        """
        path = Path(pdf_path)

        # ── Check file exists ─────────────────────────────────────────
        if not path.exists():
            logger.error(f"File not found: {pdf_path}")
            return ValidationResult(
                is_valid=False, error=f"File not found: {pdf_path}"
            )

        # ── Check file extension ──────────────────────────────────────
        if path.suffix.lower() != ".pdf":
            logger.error(f"Not a PDF file: {pdf_path}")
            return ValidationResult(
                is_valid=False, error=f"File is not a PDF: {path.suffix}"
            )

        # ── Check file size ───────────────────────────────────────────
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.max_file_size_mb:
            logger.error(
                f"File too large: {file_size_mb:.1f}MB "
                f"(max: {settings.max_file_size_mb}MB)"
            )
            return ValidationResult(
                is_valid=False,
                error=(
                    f"File size {file_size_mb:.1f}MB exceeds "
                    f"maximum of {settings.max_file_size_mb}MB"
                ),
            )

        # ── Try to parse as PDF ───────────────────────────────────────
        try:
            reader = PdfReader(pdf_path)
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            return ValidationResult(
                is_valid=False, error=f"Invalid or corrupted PDF: {e}"
            )

        # ── Check page count ──────────────────────────────────────────
        page_count = len(reader.pages)
        if page_count == 0:
            logger.error("PDF has no pages")
            return ValidationResult(
                is_valid=False, error="PDF contains no pages"
            )

        if page_count > settings.max_pages:
            logger.error(
                f"Too many pages: {page_count} (max: {settings.max_pages})"
            )
            return ValidationResult(
                is_valid=False,
                error=(
                    f"Page count {page_count} exceeds "
                    f"maximum of {settings.max_pages}"
                ),
            )

        # ── Check that at least one page has extractable text ─────────
        has_text = False
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                has_text = True
                break

        if not has_text:
            logger.error("No extractable text found in PDF")
            return ValidationResult(
                is_valid=False,
                error="PDF contains no extractable text (may be scanned/image-only)",
            )

        # ── All checks passed ─────────────────────────────────────────
        logger.info(
            f"PDF validated: {path.name} | "
            f"{page_count} pages | {file_size_mb:.1f}MB"
        )
        return ValidationResult(
            is_valid=True,
            metadata={
                "filename": path.name,
                "file_size_mb": round(file_size_mb, 2),
                "page_count": page_count,
            },
        )
