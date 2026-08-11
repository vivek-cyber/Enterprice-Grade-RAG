"""Text normalization for parsed documents."""

from __future__ import annotations

import re
import unicodedata


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HORIZONTAL_SPACE = re.compile(r"[ \t]+")
MANY_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize text while preserving paragraph and code-block readability."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = CONTROL_CHARS.sub("", normalized)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []
    for line in normalized.split("\n"):
        cleaned_lines.append(HORIZONTAL_SPACE.sub(" ", line).rstrip())

    normalized = "\n".join(cleaned_lines)
    normalized = MANY_BLANK_LINES.sub("\n\n", normalized)
    return normalized.strip()
