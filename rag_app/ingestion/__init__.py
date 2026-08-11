"""Ingestion layer for loading, parsing, cleaning, and chunking documents."""

from rag_app.ingestion.pipeline import ingest_folder

__all__ = ["ingest_folder"]
