"""Embedding interfaces and providers for the ingestion pipeline."""

from rag_app.ingestion.embeddings.base import EmbeddingProvider, EmbeddingRecord
from rag_app.ingestion.embeddings.fallback_provider import (
    FallbackEmbeddingError,
    FallbackEmbeddingProvider,
)
from rag_app.ingestion.embeddings.gemini_provider import GeminiEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "EmbeddingRecord",
    "FallbackEmbeddingError",
    "FallbackEmbeddingProvider",
    "GeminiEmbeddingProvider",
]
