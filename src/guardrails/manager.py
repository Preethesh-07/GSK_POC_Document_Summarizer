"""
Guardrail Manager — orchestrates all validation steps.

Runs PDF validation → text extraction → text validation → injection scan
in sequence. Returns an aggregated result.
"""

from dataclasses import dataclass, field

from src.guardrails.pdf_validator import PDFValidator, ValidationResult
from src.guardrails.prompt_injection import PromptInjectionDetector
from src.guardrails.text_validator import TextValidator
from src.rag.loader import PDFLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GuardrailResult:
    """Aggregated result from all guardrail checks."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    extracted_text: str = ""


class GuardrailManager:
    """Orchestrates all input guardrail validations."""

    def __init__(self):
        self.pdf_validator = PDFValidator()
        self.text_validator = TextValidator()
        self.injection_detector = PromptInjectionDetector()
        self.loader = PDFLoader()

    def validate(self, pdf_path: str) -> GuardrailResult:
        """Run the full guardrail validation pipeline.

        Args:
            pdf_path: Path to the PDF file to validate.

        Returns:
            GuardrailResult with aggregated validation outcome.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # ── Step 1: PDF file validation ───────────────────────────────
        logger.info("Running PDF file validation...")
        pdf_result: ValidationResult = self.pdf_validator.validate(pdf_path)
        if not pdf_result.is_valid:
            return GuardrailResult(
                is_valid=False,
                errors=[pdf_result.error or "PDF validation failed"],
                metadata=pdf_result.metadata,
            )

        # ── Step 2: Extract text via loader ───────────────────────────
        logger.info("Extracting text for validation...")
        try:
            pages = self.loader.load(pdf_path)
            full_text = "\n".join(page.page_content for page in pages)
        except Exception as e:
            return GuardrailResult(
                is_valid=False,
                errors=[f"Text extraction failed: {e}"],
                metadata=pdf_result.metadata,
            )

        # ── Step 3: Text quality validation ───────────────────────────
        logger.info("Running text quality validation...")
        text_result = self.text_validator.validate(full_text)
        if not text_result.is_valid:
            return GuardrailResult(
                is_valid=False,
                errors=[text_result.error or "Text validation failed"],
                metadata=pdf_result.metadata,
            )

        # ── Step 4: Prompt injection scan ─────────────────────────────
        logger.info("Running prompt injection scan...")
        injection_result = self.injection_detector.scan(full_text)
        if not injection_result.is_safe:
            # Injection patterns are warnings, not hard failures
            # (documents may contain legitimate text matching patterns)
            warnings.extend(injection_result.findings)
            logger.warning(
                f"Prompt injection warnings: {len(injection_result.findings)}"
            )

        # ── All checks passed ─────────────────────────────────────────
        logger.info("All guardrail checks passed")
        return GuardrailResult(
            is_valid=True,
            errors=errors,
            warnings=warnings,
            metadata=pdf_result.metadata,
            extracted_text=full_text,
        )
