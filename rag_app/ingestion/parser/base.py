"""Base parser interface and helpers."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from rag_app.ingestion.models import Document, ParseResult


class BaseParser(ABC):
    """Abstract parser for one or more source file extensions."""

    parser_name: str
    supported_extensions: set[str]

    @abstractmethod
    def parse(self, path: Path) -> ParseResult:
        """Parse a file into a normalized document."""

    def make_document(
        self,
        path: Path,
        content: str,
        *,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> Document:
        source_type = path.suffix.lower()
        return Document(
            id=stable_document_id(path),
            source_path=path,
            source_type=source_type,
            title=title or path.stem,
            content=content,
            metadata={
                "source_name": path.name,
                "source_extension": source_type,
                **(metadata or {}),
            },
        )


def stable_document_id(path: Path) -> str:
    """Create a deterministic document id from the resolved source path."""

    normalized = str(path.resolve()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
