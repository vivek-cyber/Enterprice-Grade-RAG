"""Deterministic text chunking."""

from __future__ import annotations

import hashlib

from rag_app.ingestion.models import Document, DocumentChunk


def chunk_document(
    document: Document,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[DocumentChunk]:
    """Split a document into stable character-aware chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = document.content.strip()
    if not text:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = _find_boundary(text, start, end)
            if boundary > start:
                end = boundary

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                DocumentChunk(
                    id=_stable_chunk_id(document.id, chunk_index, chunk_text),
                    document_id=document.id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    metadata={
                        **document.metadata,
                        "source_path": str(document.source_path),
                        "source_type": document.source_type,
                        "chunk_start": start,
                        "chunk_end": end,
                    },
                )
            )
            chunk_index += 1

        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def _find_boundary(text: str, start: int, end: int) -> int:
    window = text[start:end]
    for separator in ("\n\n", "\n", ". ", " "):
        index = window.rfind(separator)
        if index >= int(len(window) * 0.5):
            return start + index + len(separator)
    return end


def _stable_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(
        f"{document_id}:{chunk_index}:{text}".encode("utf-8")
    ).hexdigest()
    return digest
