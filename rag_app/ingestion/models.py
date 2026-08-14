"""Shared data contracts for the ingestion layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_app.ingestion.embeddings.base import EmbeddingRecord

Metadata = dict[str, Any]


@dataclass(slots=True)
class Document:
    """A parsed source document before chunking."""

    id: str
    source_path: Path
    source_type: str
    title: str | None
    content: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class DocumentChunk:
    """A deterministic chunk derived from a parsed document."""

    id: str
    document_id: str
    text: str
    chunk_index: int
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class ParseResult:
    """Parser output for one source file."""

    document: Document | None
    parser_name: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.document is not None and not self.errors


@dataclass(slots=True)
class FileRecord:
    """Discovered source file plus stable identity metadata."""

    path: Path
    extension: str
    sha256: str
    size_bytes: int
    supported: bool
    skip_reason: str | None = None


@dataclass(slots=True)
class IngestionReport:
    """End-to-end result of an ingestion run."""

    source_dir: Path
    total_files: int = 0
    parsed_files: int = 0
    skipped_files: list[FileRecord] = field(default_factory=list)
    failed_files: dict[Path, list[str]] = field(default_factory=dict)
    documents: list[Document] = field(default_factory=list)
    chunks: list[DocumentChunk] = field(default_factory=list)
    embeddings: list[EmbeddingRecord] = field(default_factory=list)
    warnings: dict[Path, list[str]] = field(default_factory=dict)
    parser_timings_ms: dict[Path, float] = field(default_factory=dict)

    @property
    def failed_count(self) -> int:
        return len(self.failed_files)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_files)
