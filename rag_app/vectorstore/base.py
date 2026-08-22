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


@dataclass(slots=True)
class MetadataFilter:
    """A store-agnostic restriction on which points a search may return.

    Deliberately minimal: exact match and match-any cover every payload field
    ingestion actually writes (source_name, source_path, source_type,
    source_extension). Richer predicates -- ranges, substring match, negation
    -- are left out until a caller needs one, since each has to be expressible
    by every VectorStore implementation, not just Qdrant.

    All conditions combine with AND: a point must satisfy every entry in
    `equals` and every entry in `any_of` to be returned.
    """

    # field name -> the single value it must equal
    equals: dict[str, Any] = field(default_factory=dict)
    # field name -> values, any one of which it may equal
    any_of: dict[str, list[Any]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Report whether this filter would restrict anything at all."""

        return not self.equals and not self.any_of


class VectorStore(Protocol):
    """Something that can index and search embedded chunks."""

    store_name: str

    def ensure_collection(self, vector_size: int) -> None:
        """Create the backing collection if it doesn't already exist."""

    def upsert(self, points: list[VectorPoint]) -> None:
        """Insert or overwrite points, keyed by chunk_id."""

    def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        filters: MetadataFilter | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorMatch]:
        """Return the nearest points to vector, most similar first.

        filters restricts which points are eligible; score_threshold drops
        matches scoring below it, which is how a caller says "return nothing
        rather than return junk".
        """

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove points by chunk_id."""
