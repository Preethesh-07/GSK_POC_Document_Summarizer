"""
Prompt injection detection.

Scans extracted text for known prompt injection patterns
that could manipulate LLM behavior.
"""

import re
from dataclasses import dataclass, field

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InjectionScanResult:
    """Result of prompt injection scan."""

    is_safe: bool
    findings: list[str] = field(default_factory=list)


class PromptInjectionDetector:
    """Detects potential prompt injection patterns in document text."""

    # Known prompt injection patterns (case-insensitive)
    INJECTION_PATTERNS: list[str] = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?above\s+instructions",
        r"disregard\s+(all\s+)?previous",
        r"forget\s+(all\s+)?previous",
        r"you\s+are\s+now\s+a",
        r"act\s+as\s+(a|an)\s+",
        r"new\s+system\s+prompt",
        r"override\s+system\s+prompt",
        r"system\s*:\s*you\s+are",
        r"\[system\]",
        r"\[inst\]",
        r"<\s*system\s*>",
        r"do\s+not\s+summarize",
        r"instead\s+of\s+summarizing",
        r"reveal\s+(your|the)\s+(system\s+)?prompt",
        r"print\s+(your|the)\s+(system\s+)?prompt",
        r"output\s+(your|the)\s+instructions",
    ]

    def __init__(self):
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.INJECTION_PATTERNS
        ]

    def scan(self, text: str) -> InjectionScanResult:
        """Scan text for prompt injection patterns.

        Args:
            text: Extracted document text to scan.

        Returns:
            InjectionScanResult with is_safe flag and list of findings.
        """
        findings: list[str] = []

        for pattern in self._compiled_patterns:
            matches = pattern.findall(text)
            if matches:
                finding = (
                    f"Injection pattern detected: '{pattern.pattern}' "
                    f"({len(matches)} occurrence(s))"
                )
                findings.append(finding)
                logger.warning(finding)

        if findings:
            logger.warning(
                f"Prompt injection scan: {len(findings)} pattern(s) detected"
            )
            return InjectionScanResult(is_safe=False, findings=findings)

        logger.info("Prompt injection scan: clean")
        return InjectionScanResult(is_safe=True)
