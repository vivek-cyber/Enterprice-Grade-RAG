"""Vector store contracts.

Concrete stores (e.g. qdrant_store.QdrantVectorStore) live alongside this
module. Callers depend only on this Protocol so the underlying store can be
swapped without touching ingestion or retrieval code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class VectorPoint:
    """One embedded chunk ready to be indexed."""

    chunk_id: str
    vector: list[float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VectorMatch:
    """One search result returned from a vector store query."""

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    """Contract implemented by every vector store backend."""

    store_name: str

    def ensure_collection(self, vector_size: int) -> None:
        """Create the backing collection if it doesn't already exist."""

    def upsert(self, points: list[VectorPoint]) -> None:
        """Insert or overwrite points, keyed by chunk_id."""

    def search(self, vector: list[float], *, limit: int = 10) -> list[VectorMatch]:
        """Return the nearest points to vector, most similar first."""

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove points by chunk_id."""
