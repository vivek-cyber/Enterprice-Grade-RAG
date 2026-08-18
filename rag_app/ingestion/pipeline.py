"""Ingestion pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from rag_app.ingestion.chunking import chunk_document
from rag_app.ingestion.cleaners import clean_text
from rag_app.ingestion.embeddings.base import EmbeddingProvider
from rag_app.ingestion.file_discovery import discover_files
from rag_app.ingestion.models import IngestionReport
from rag_app.ingestion.parser import build_parser_registry
from rag_app.ingestion.parser.base import BaseParser
from rag_app.vectorstore.base import VectorPoint, VectorStore


def ingest_folder(
    source_dir: Path,
    *,
    include_globs: list[str] | None = None,
    parsers: tuple[BaseParser, ...] | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
) -> IngestionReport:
    """Discover, parse, clean, and chunk supported files in a folder."""

    if vector_store is not None and embedding_provider is None:
        raise ValueError("vector_store requires an embedding_provider")

    registry = build_parser_registry(parsers)
    supported_extensions = set(registry)
    records = discover_files(
        source_dir,
        supported_extensions,
        include_globs=include_globs,
    )
    report = IngestionReport(source_dir=source_dir.resolve(), total_files=len(records))

    for record in records:
        if not record.supported:
            report.skipped_files.append(record)
            continue

        parser = registry[record.extension]
        started = perf_counter()
        try:
            result = parser.parse(record.path)
        except Exception as exc:
            result = None
            report.failed_files[record.path] = [f"Unexpected parser failure: {exc}"]
        finally:
            elapsed_ms = (perf_counter() - started) * 1000
            report.parser_timings_ms[record.path] = elapsed_ms

        if result is None:
            continue
        if result.warnings:
            report.warnings[record.path] = result.warnings
        if result.errors or result.document is None:
            report.failed_files[record.path] = result.errors or ["Parser returned no document"]
            continue

        result.document.content = clean_text(result.document.content)
        document_chunks = chunk_document(
            result.document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        report.documents.append(result.document)
        report.chunks.extend(document_chunks)
        report.parsed_files += 1

    if embedding_provider is not None and report.chunks:
        chunk_texts = [chunk.text for chunk in report.chunks]
        embedding_records = embedding_provider.embed_texts(chunk_texts)
        for chunk, record in zip(report.chunks, embedding_records, strict=True):
            record.chunk_id = chunk.id
        report.embeddings = embedding_records

        if vector_store is not None:
            vector_store.ensure_collection(len(embedding_records[0].vector))
            points = [
                VectorPoint(
                    chunk_id=chunk.id,
                    vector=record.vector,
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
                for chunk, record in zip(report.chunks, embedding_records, strict=True)
            ]
            vector_store.upsert(points)

    return report
