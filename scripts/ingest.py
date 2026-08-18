"""CLI entrypoint for running the ingestion pipeline against a data folder.

Embeddings run locally via Nomic Embed Text v1.5; no API key or network
access is required for the embedding step.

Usage:
    python scripts/ingest.py DATA/true_data --collection rag_documents
    python scripts/ingest.py DATA --dry-run   # parse+chunk only, no embedding/upsert
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import logfire
from dotenv import load_dotenv

# Running this file directly puts scripts/ on sys.path, not the project root, so
# `rag_app` would not resolve. The package is not pip-installable yet (no
# pyproject.toml), so add the repo root explicitly instead of relying on the
# caller to export PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_app.ingestion.checkpoint import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIRNAME,
    ChunkCheckpointStore,
)
from rag_app.ingestion.embeddings.nomic_provider import NomicEmbeddingProvider  # noqa: E402
from rag_app.ingestion.pipeline import ingest_folder  # noqa: E402
from rag_app.vectorstore.qdrant_store import QdrantVectorStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("source_dir", type=Path, help="Folder to ingest (searched recursively).")
    parser.add_argument("--collection", default="rag_documents", help="Qdrant collection name.")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk only; skip embedding generation and vector store writes.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-file parse/chunk checkpoints "
            f"(default: {DEFAULT_CHECKPOINT_DIRNAME} in the project root)."
        ),
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing; reparse every file even if cached.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    logfire.configure(service_name="rag-ingestion", send_to_logfire="if-token-present")
    args = parse_args()

    if not args.source_dir.exists():
        print(f"Source directory does not exist: {args.source_dir}", file=sys.stderr)
        return 1

    embedding_provider = None
    vector_store = None
    if not args.dry_run:
        embedding_provider = NomicEmbeddingProvider()
        vector_store = QdrantVectorStore(collection_name=args.collection)

    checkpoint_store = None
    if not args.no_checkpoint:
        checkpoint_dir = args.checkpoint_dir or PROJECT_ROOT / DEFAULT_CHECKPOINT_DIRNAME
        checkpoint_store = ChunkCheckpointStore(checkpoint_dir)

    report = ingest_folder(
        args.source_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        checkpoint_store=checkpoint_store,
    )

    total_chunk_chars = sum(len(chunk.text) for chunk in report.chunks)
    print(f"Source dir:      {report.source_dir}")
    print(f"Total files:     {report.total_files}")
    print(f"Parsed files:    {report.parsed_files} ({report.cached_files} from checkpoint)")
    print(f"Skipped files:   {report.skipped_count}")
    print(f"Failed files:    {report.failed_count}")
    print(f"Degraded files:  {len(report.warnings)}")
    print(f"Chunks produced: {len(report.chunks)}")
    print(f"Chunk text size: {total_chunk_chars / 1_000_000:.2f} MB")
    if report.embeddings:
        print(f"Embeddings:      {len(report.embeddings)}")
        print(f"Embedding model: {report.embeddings[0].model}")
        print(f"Vector dim:      {len(report.embeddings[0].vector)}")

    if report.failed_files:
        print("\nFailed files:")
        for path, errors in report.failed_files.items():
            print(f"  {path}: {errors}")

    if report.warnings:
        # Degraded parses still produce chunks, so they never show up as
        # failures -- surface them explicitly or a silently truncated document
        # sails into the index looking healthy.
        print("\nDegraded files (parsed with warnings):")
        for path, warnings in report.warnings.items():
            head = "; ".join(warnings[:2])
            extra = f" (+{len(warnings) - 2} more)" if len(warnings) > 2 else ""
            print(f"  {path.name}: {head}{extra}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
