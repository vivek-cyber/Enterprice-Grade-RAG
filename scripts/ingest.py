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

from rag_app.ingestion.embeddings.nomic_provider import NomicEmbeddingProvider
from rag_app.ingestion.pipeline import ingest_folder
from rag_app.vectorstore.qdrant_store import QdrantVectorStore


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

    report = ingest_folder(
        args.source_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    total_chunk_chars = sum(len(chunk.text) for chunk in report.chunks)
    print(f"Source dir:      {report.source_dir}")
    print(f"Total files:     {report.total_files}")
    print(f"Parsed files:    {report.parsed_files}")
    print(f"Skipped files:   {report.skipped_count}")
    print(f"Failed files:    {report.failed_count}")
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
