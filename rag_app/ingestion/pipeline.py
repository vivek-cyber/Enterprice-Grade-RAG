"""Ingestion pipeline orchestration."""

from __future__ import annotations

from bisect import bisect_right
from itertools import accumulate
from pathlib import Path
from time import perf_counter

import logfire

from rag_app.ingestion.checkpoint import CheckpointEntry, ChunkCheckpointStore
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
    checkpoint_store: ChunkCheckpointStore | None = None,
) -> IngestionReport:
    """Discover, parse, clean, and chunk supported files in a folder.

    When a checkpoint_store is supplied, each file's chunks are persisted as
    soon as that file is chunked and replayed from cache on later runs, so an
    interrupted run does not have to reparse work it already finished.
    """

    if vector_store is not None and embedding_provider is None:
        raise ValueError("vector_store requires an embedding_provider")

    with logfire.span(
        "ingest_folder",
        source_dir=str(source_dir),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ) as run_span:
        registry = build_parser_registry(parsers)
        supported_extensions = set(registry)
        records = discover_files(
            source_dir,
            supported_extensions,
            include_globs=include_globs,
        )
        report = IngestionReport(source_dir=source_dir.resolve(), total_files=len(records))
        run_span.set_attribute("total_files", report.total_files)

        supported_total = sum(1 for record in records if record.supported)
        # Chunk count per successfully parsed document, in the same order the
        # chunks were appended to report.chunks. Lets the embedding step map a
        # running chunk count back to "how many files are fully embedded".
        chunks_per_document: list[int] = []
        file_index = 0

        for record in records:
            if not record.supported:
                report.skipped_files.append(record)
                continue

            file_index += 1

            cache_key = (
                checkpoint_store.key(
                    record, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                if checkpoint_store is not None
                else None
            )
            if checkpoint_store is not None and cache_key is not None:
                cached = checkpoint_store.load(cache_key)
                if cached is not None:
                    report.documents.append(cached.document)
                    report.chunks.extend(cached.chunks)
                    chunks_per_document.append(len(cached.chunks))
                    report.parsed_files += 1
                    report.cached_files += 1
                    if cached.warnings:
                        report.warnings[record.path] = cached.warnings
                    logfire.info(
                        "cached {file_index}/{supported_total}: {filename} "
                        "({chunks} chunks, {files_left} left)",
                        file_index=file_index,
                        supported_total=supported_total,
                        filename=record.path.name,
                        chunks=len(cached.chunks),
                        files_left=supported_total - file_index,
                    )
                    continue

            parser = registry[record.extension]
            started = perf_counter()
            with logfire.span(
                "parse {file_index}/{supported_total}: {filename} ({files_left} left)",
                file_index=file_index,
                supported_total=supported_total,
                filename=record.path.name,
                files_left=supported_total - file_index,
                file=str(record.path),
                parser=parser.__class__.__name__,
                size_bytes=record.size_bytes,
            ) as file_span:
                try:
                    result = parser.parse(record.path)
                except Exception as exc:
                    result = None
                    report.failed_files[record.path] = [f"Unexpected parser failure: {exc}"]
                    file_span.set_attribute("failed", True)
                finally:
                    elapsed_ms = (perf_counter() - started) * 1000
                    report.parser_timings_ms[record.path] = elapsed_ms
                    file_span.set_attribute("elapsed_ms", elapsed_ms)

                if result is None:
                    continue
                if result.warnings:
                    report.warnings[record.path] = result.warnings
                    file_span.set_attribute("warnings", result.warnings)
                if result.errors or result.document is None:
                    report.failed_files[record.path] = result.errors or [
                        "Parser returned no document"
                    ]
                    file_span.set_attribute("failed", True)
                    continue

                result.document.content = clean_text(result.document.content)
                document_chunks = chunk_document(
                    result.document,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                report.documents.append(result.document)
                report.chunks.extend(document_chunks)
                chunks_per_document.append(len(document_chunks))
                report.parsed_files += 1
                file_span.set_attribute("chunks_produced", len(document_chunks))

                if checkpoint_store is not None and cache_key is not None:
                    # Persist as soon as this file is chunked: a later crash or
                    # kill then costs only the file in flight, not the run.
                    try:
                        checkpoint_store.save(
                            cache_key,
                            CheckpointEntry(
                                document=result.document,
                                chunks=document_chunks,
                                warnings=result.warnings,
                            ),
                        )
                    except OSError as exc:
                        # Checkpointing is an optimization; never fail the run
                        # over a cache write.
                        logfire.warn(
                            "checkpoint write failed for {file}: {error}",
                            file=str(record.path),
                            error=str(exc),
                        )

        logfire.info(
            "parsed source folder",
            total_files=report.total_files,
            parsed_files=report.parsed_files,
            cached_files=report.cached_files,
            skipped_files=report.skipped_count,
            failed_files=report.failed_count,
            degraded_files=len(report.warnings),
            chunks_produced=len(report.chunks),
        )

        if embedding_provider is not None and report.chunks:
            chunk_texts = [chunk.text for chunk in report.chunks]
            # Cumulative chunk index at which each document becomes fully
            # embedded, e.g. [12, 30, 45] for three documents.
            document_boundaries = list(accumulate(chunks_per_document))
            document_total = len(document_boundaries)
            with logfire.span(
                "embed_chunks",
                provider=embedding_provider.provider_name,
                model=embedding_provider.model_name,
                chunk_count=len(chunk_texts),
                file_count=document_total,
            ):
                def report_embedding_progress(completed: int, total: int) -> None:
                    files_done = bisect_right(document_boundaries, completed)
                    logfire.info(
                        "embedded {files_done}/{file_total} files, "
                        "{completed}/{total} chunks ({remaining} left)",
                        files_done=files_done,
                        file_total=document_total,
                        completed=completed,
                        total=total,
                        remaining=total - completed,
                    )

                embedding_records = embedding_provider.embed_texts(
                    chunk_texts,
                    progress_callback=report_embedding_progress,
                )
            for chunk, record in zip(report.chunks, embedding_records, strict=True):
                record.chunk_id = chunk.id
            report.embeddings = embedding_records

            if vector_store is not None:
                with logfire.span(
                    "upsert_vectors",
                    store=vector_store.store_name,
                    point_count=len(embedding_records),
                ):
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

        run_span.set_attribute("parsed_files", report.parsed_files)
        run_span.set_attribute("failed_files", report.failed_count)
        run_span.set_attribute("chunks_produced", len(report.chunks))
        run_span.set_attribute("embeddings_produced", len(report.embeddings))

    return report
