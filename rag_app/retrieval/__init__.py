"""Retrieval interfaces and implementations for querying the vector store."""

from rag_app.retrieval.base import (
    DEFAULT_LIMIT,
    RetrievalResult,
    RetrievedChunk,
    Retriever,
)
from rag_app.retrieval.dense import (
    DenseRetriever,
    QueryPrefixError,
    build_query_embedder,
)

__all__ = [
    "DEFAULT_LIMIT",
    "DenseRetriever",
    "QueryPrefixError",
    "RetrievalResult",
    "RetrievedChunk",
    "Retriever",
    "build_query_embedder",
]
