"""CLI entrypoint for querying the vector store with dense retrieval.

The query is embedded locally with Nomic Embed Text v1.5 (same model as
ingestion, but with the query task prefix), so no API key or network access
is needed beyond reaching Qdrant.

Usage:
    python scripts/query.py "how does copy-on-write work"
    python scripts/query.py "cache coherence" --limit 5 --source-type .pdf
    python scripts/query.py "malloc" --full --min-score 0.5
    python scripts/query.py "paging" --json > hits.json
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
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

from rag_app.retrieval import (  # noqa: E402
    DEFAULT_LIMIT,
    DenseRetriever,
    RetrievalResult,
    build_query_embedder,
)
from rag_app.vectorstore.base import MetadataFilter  # noqa: E402
from rag_app.vectorstore.qdrant_store import QdrantVectorStore  # noqa: E402

SNIPPET_CHARS = 400


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a single retrieval query."""

    parser = argparse.ArgumentParser(
        description="Query the vector store and print ranked chunks."
    )
    parser.add_argument("query", help="The search query.")
    parser.add_argument(
        "--collection", default="rag_documents", help="Qdrant collection name."
    )
    parser.add_argument(
        "-k",
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"How many chunks to return (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help=(
            "Drop matches scoring below this cosine similarity (0-1). Returns "
            "fewer results -- or none -- rather than padding with weak matches."
        ),
    )
    parser.add_argument(
        "--source-type",
        action="append",
        default=None,
        metavar="EXT",
        help="Only search files with this extension, e.g. .pdf. Repeatable.",
    )
    parser.add_argument(
        "--source-name",
        action="append",
        default=None,
        metavar="FILENAME",
        help="Only search this exact filename. Repeatable.",
    )
    parser.add_argument(
        "--max-per-document",
        type=int,
        default=None,
        help=(
            "Cap chunks returned from any one file (default: 3). Pass 0 to "
            "disable and allow one document to fill every slot."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"Print each chunk in full instead of the first {SNIPPET_CHARS} characters.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON for piping into other tools.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show Logfire spans (timing, candidate counts) on the console.",
    )
    return parser.parse_args()


def build_filters(args: argparse.Namespace) -> MetadataFilter | None:
    """Translate the filter flags into a MetadataFilter, or None if unused."""

    any_of: dict[str, list[str]] = {}
    if args.source_type:
        # Normalize `pdf` to `.pdf` so both spellings work at the CLI.
        any_of["source_type"] = [
            ext if ext.startswith(".") else f".{ext}" for ext in args.source_type
        ]
    if args.source_name:
        any_of["source_name"] = list(args.source_name)

    filters = MetadataFilter(any_of=any_of)
    return None if filters.is_empty() else filters


def print_results(result: RetrievalResult, *, full: bool) -> None:
    """Print ranked chunks as readable text."""

    if result.is_empty:
        print(f'No matches for "{result.query}".')
        if result.candidates_considered == 0:
            print(
                "The store returned nothing at all -- check the collection "
                "name, or lower --min-score."
            )
        return

    print(f'Query: "{result.query}"')
    print(f"Returned {len(result)} of {result.candidates_considered} candidates.\n")

    for chunk in result.chunks:
        header = f"[{chunk.rank}] score={chunk.score:.4f}  {chunk.source_name}"
        print(header)
        print("-" * min(len(header), 100))
        body = (
            chunk.text
            if full
            else textwrap.shorten(chunk.text, width=SNIPPET_CHARS, placeholder=" ...")
        )
        print(textwrap.indent(body, "    "))
        print()

    if result.dropped_by_document_cap:
        print(
            f"({result.dropped_by_document_cap} further match(es) hidden by the "
            "per-document cap; raise --max-per-document to see them.)"
        )


def print_json(result: RetrievalResult) -> None:
    """Print ranked chunks as a JSON document on stdout."""

    print(
        json.dumps(
            {
                "query": result.query,
                "retriever": result.retriever_name,
                "candidates_considered": result.candidates_considered,
                "dropped_by_document_cap": result.dropped_by_document_cap,
                "results": [
                    {
                        "rank": chunk.rank,
                        "score": chunk.score,
                        "chunk_id": chunk.chunk_id,
                        "source_name": chunk.source_name,
                        "source_path": chunk.source_path,
                        "text": chunk.text,
                    }
                    for chunk in result.chunks
                ],
            },
            indent=2,
            default=str,
        )
    )


def main() -> int:
    """Run one retrieval query and print the results."""

    # Chunk text comes out of PDFs full of box-drawing, dashes, and ligatures
    # that the default Windows console codepage (cp1252) cannot encode, which
    # otherwise crashes mid-way through printing results.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    load_dotenv()
    args = parse_args()
    # Spans are noise around a handful of search results, so the console sink
    # is off unless asked for. --json in particular must keep stdout parseable.
    logfire.configure(
        service_name="rag-retrieval",
        send_to_logfire="if-token-present",
        console=logfire.ConsoleOptions() if args.verbose else False,
    )

    if args.limit < 1:
        print("--limit must be at least 1", file=sys.stderr)
        return 1

    retriever = DenseRetriever(
        embedding_provider=build_query_embedder(),
        vector_store=QdrantVectorStore(collection_name=args.collection),
        score_threshold=args.min_score,
        **(
            {"max_per_document": args.max_per_document or None}
            if args.max_per_document is not None
            else {}
        ),
    )

    result = retriever.retrieve(
        args.query,
        limit=args.limit,
        filters=build_filters(args),
    )

    if args.json:
        print_json(result)
    else:
        print_results(result, full=args.full)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
