"""CLI entrypoint for running the ingestion pipeline against a data folder.

Embeddings run locally via Nomic Embed Text v1.5; no API key or network
access is required for the embedding step.

Usage:
    python scripts/ingest.py DATA/true_data --collection rag_documents
    python scripts/ingest.py DATA --dry-run   # parse+chunk only, no embedding/upsert
    python scripts/ingest.py DATA --auto-resume  # restart on crash until fully ingested
"""

from __future__ import annotations

import argparse
import subprocess
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
    """Parse CLI arguments, rejecting the invalid --auto-resume + --no-checkpoint combo."""

    parser = argparse.ArgumentParser(
        description="Ingest documents into the vector store."
    )
    parser.add_argument(
        "source_dir", type=Path, help="Folder to ingest (searched recursively)."
    )
    parser.add_argument(
        "--collection", default="rag_documents", help="Qdrant collection name."
    )
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
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help=(
            "Restart ingestion in a fresh process whenever it crashes, until "
            "every file is parsed. Docling's PDF backend leaks memory across "
            "files within one process and eventually dies on a large corpus; "
            "each fresh process starts clean, and finished files are skipped "
            "via the checkpoint cache. Requires checkpointing to be enabled."
        ),
    )
    parser.add_argument(
        "--max-resume-attempts",
        type=int,
        default=20,
        help="With --auto-resume, give up after this many restarts (default: 20).",
    )
    args = parser.parse_args()

    if args.auto_resume and args.no_checkpoint:
        parser.error("--auto-resume requires checkpointing; drop --no-checkpoint.")

    return args


def run_auto_resume(args: argparse.Namespace) -> int:
    """Rerun this script in a fresh subprocess each time it crashes.

    Docling's memory leak accumulates within one process, not across
    processes, so a crashed attempt is not wasted: every file it finished
    before dying is already durable in the checkpoint cache (see
    ChunkCheckpointStore), and the next attempt skips straight past those
    files. Progress is measured by counting checkpoint entries rather than
    parsing subprocess output, so it stays correct even if log formatting
    changes later.
    """

    checkpoint_dir = args.checkpoint_dir or PROJECT_ROOT / DEFAULT_CHECKPOINT_DIRNAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    child_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.source_dir),
        "--collection",
        args.collection,
        "--chunk-size",
        str(args.chunk_size),
        "--chunk-overlap",
        str(args.chunk_overlap),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if args.dry_run:
        child_argv.append("--dry-run")

    def cached_file_count() -> int:
        return sum(1 for _ in checkpoint_dir.glob("*.json"))

    for attempt in range(1, args.max_resume_attempts + 1):
        before = cached_file_count()
        print(
            f"\n=== auto-resume: attempt {attempt}/{args.max_resume_attempts} "
            f"({before} file(s) already cached) ==="
        )

        # Inherits this process's stdio so the child's own Logfire/print output
        # streams live, exactly as if it had been run directly.
        result = subprocess.run(child_argv)

        if result.returncode == 0:
            print(f"\n=== auto-resume: completed on attempt {attempt} ===")
            return 0

        after = cached_file_count()
        print(
            f"=== auto-resume: attempt {attempt} exited with code "
            f"{result.returncode} ({before} -> {after} files cached) ==="
        )
        if after <= before:
            print(
                "=== auto-resume: no new files were cached on this attempt; "
                "stopping instead of retrying forever ===",
                file=sys.stderr,
            )
            return 1

    print(
        f"=== auto-resume: gave up after {args.max_resume_attempts} attempts ===",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    """Run ingestion (or --auto-resume's restart loop) and print a summary report."""

    load_dotenv()
    logfire.configure(service_name="rag-ingestion", send_to_logfire="if-token-present")
    args = parse_args()

    if not args.source_dir.exists():
        print(f"Source directory does not exist: {args.source_dir}", file=sys.stderr)
        return 1

    if args.auto_resume:
        return run_auto_resume(args)

    embedding_provider = None
    vector_store = None
    if not args.dry_run:
        embedding_provider = NomicEmbeddingProvider()
        vector_store = QdrantVectorStore(collection_name=args.collection)

    checkpoint_store = None
    if not args.no_checkpoint:
        checkpoint_dir = (
            args.checkpoint_dir or PROJECT_ROOT / DEFAULT_CHECKPOINT_DIRNAME
        )
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
    print(
        f"Parsed files:    {report.parsed_files} ({report.cached_files} from checkpoint)"
    )
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
