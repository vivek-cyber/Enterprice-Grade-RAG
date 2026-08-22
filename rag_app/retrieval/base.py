"""Retrieval contracts.

Concrete retrievers (e.g. dense.DenseRetriever) live alongside this module.
Callers -- a CLI, a Streamlit page, a LangGraph node -- depend only on the
Retriever Protocol, so a dense retriever can later be swapped for a hybrid or
reranking one without any caller changing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_app.vectorstore.base import MetadataFilter

DEFAULT_LIMIT = 10


@dataclass(slots=True)
class RetrievedChunk:
    """One chunk returned for a query, in final ranked order."""

    chunk_id: str
    # 1-based position in the returned list, so callers can cite "result 3"
    # without recomputing it from the list index.
    rank: int
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_name(self) -> str:
        """Return the originating filename, or a placeholder if absent."""

        return str(self.metadata.get("source_name", "<unknown source>"))

    @property
    def source_path(self) -> str:
        """Return the originating file path, or an empty string if absent."""

        return str(self.metadata.get("source_path", ""))


@dataclass(slots=True)
class RetrievalResult:
    """Everything one retrieval call produced, including how it got there.

    The diagnostic counts are not decoration: with over-fetching and
    per-document capping in play, "why did I only get 4 results back when I
    asked for 10" is otherwise unanswerable without re-running the query.
    """

    query: str
    chunks: list[RetrievedChunk]
    retriever_name: str
    # Matches the store returned before dedupe/capping trimmed them.
    candidates_considered: int = 0
    # Matches dropped because their document had already filled its quota.
    dropped_by_document_cap: int = 0

    def __len__(self) -> int:
        """Report how many chunks were returned."""

        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        """Report whether the query returned nothing at all."""

        return not self.chunks


class Retriever(Protocol):
    """Something that turns a query into ranked chunks from an index."""

    retriever_name: str

    def retrieve(
        self,
        query: str,
        *,
        limit: int = DEFAULT_LIMIT,
        filters: MetadataFilter | None = None,
    ) -> RetrievalResult:
        """Return the chunks most relevant to query, best first."""
