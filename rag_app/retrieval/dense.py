"""Dense (embedding similarity) retrieval over a vector store."""

from __future__ import annotations

from dataclasses import dataclass, field

import logfire

from rag_app.ingestion.embeddings.base import EmbeddingProvider
from rag_app.ingestion.embeddings.nomic_provider import (
    DEFAULT_DOCUMENT_PREFIX,
    DEFAULT_QUERY_PREFIX,
    NomicEmbeddingProvider,
)
from rag_app.retrieval.base import (
    DEFAULT_LIMIT,
    RetrievalResult,
    RetrievedChunk,
)
from rag_app.vectorstore.base import MetadataFilter, VectorStore

# Ask the store for this multiple of the caller's limit, so per-document
# capping still has spare candidates to promote instead of returning short.
DEFAULT_OVERSAMPLE_FACTOR = 4
# Chunks from one document past this many are dropped. Without a cap, one long
# PDF whose sections all match can occupy every slot and crowd out the other
# documents that answer the question.
DEFAULT_MAX_PER_DOCUMENT = 3
# Payload field identifying the source document. Ingestion does not currently
# write document_id into the payload, so the source path stands in as document
# identity -- it is present on every point and unique per file.
DEFAULT_DOCUMENT_KEY = "source_path"


class QueryPrefixError(RuntimeError):
    """Raised when a query would be embedded with the document task prefix."""


def build_query_embedder(**overrides: object) -> NomicEmbeddingProvider:
    """Build a Nomic provider configured for *queries* rather than documents.

    Nomic Embed is prefix-conditioned: the same text embedded as a document
    and as a query lands in different places, and mixing the two silently
    wrecks relevance without raising anything. Ingestion uses the document
    prefix, so retrieval must use the query prefix -- this helper exists so
    callers never have to remember that.
    """

    overrides.setdefault("task_prefix", DEFAULT_QUERY_PREFIX)
    return NomicEmbeddingProvider(**overrides)  # type: ignore[arg-type]


@dataclass(slots=True, kw_only=True)
class DenseRetriever:
    """Embeds the query and returns the nearest chunks from a vector store."""

    embedding_provider: EmbeddingProvider
    vector_store: VectorStore
    oversample_factor: int = DEFAULT_OVERSAMPLE_FACTOR
    max_per_document: int | None = DEFAULT_MAX_PER_DOCUMENT
    document_key: str = DEFAULT_DOCUMENT_KEY
    score_threshold: float | None = None
    retriever_name: str = field(default="dense", init=False)

    def __post_init__(self) -> None:
        """Reject settings that would silently return bad or empty results."""

        if self.oversample_factor < 1:
            raise ValueError("oversample_factor must be at least 1")
        if self.max_per_document is not None and self.max_per_document < 1:
            raise ValueError("max_per_document must be at least 1, or None to disable")

        # The one mistake that produces plausible-looking but badly ranked
        # results with no error anywhere. Caught here rather than trusted to
        # the caller, because nothing downstream can detect it.
        prefix = getattr(self.embedding_provider, "task_prefix", None)
        if prefix == DEFAULT_DOCUMENT_PREFIX:
            raise QueryPrefixError(
                "Query embedder is configured with the document task prefix "
                f"({DEFAULT_DOCUMENT_PREFIX!r}). Nomic Embed is "
                "prefix-conditioned, so this degrades relevance without "
                "failing. Use rag_app.retrieval.build_query_embedder(), or "
                f"pass task_prefix={DEFAULT_QUERY_PREFIX!r}."
            )

    def retrieve(
        self,
        query: str,
        *,
        limit: int = DEFAULT_LIMIT,
        filters: MetadataFilter | None = None,
    ) -> RetrievalResult:
        """Return the chunks most similar to query, best first."""

        if limit < 1:
            raise ValueError("limit must be at least 1")

        text = query.strip()
        if not text:
            raise ValueError("query must not be empty")

        with logfire.span(
            "dense_retrieve",
            query=text,
            limit=limit,
            filtered=filters is not None and not filters.is_empty(),
        ) as span:
            vector = self._embed_query(text)
            fetch_limit = limit * self.oversample_factor
            matches = self.vector_store.search(
                vector,
                limit=fetch_limit,
                filters=filters,
                score_threshold=self.score_threshold,
            )

            chunks, dropped = self._cap_per_document(matches, limit)
            span.set_attribute("candidates", len(matches))
            span.set_attribute("returned", len(chunks))
            span.set_attribute("dropped_by_document_cap", dropped)

        return RetrievalResult(
            query=text,
            chunks=chunks,
            retriever_name=self.retriever_name,
            candidates_considered=len(matches),
            dropped_by_document_cap=dropped,
        )

    def _embed_query(self, text: str) -> list[float]:
        records = self.embedding_provider.embed_texts([text])
        if not records:
            raise RuntimeError(
                f"Embedding provider {self.embedding_provider.provider_name} "
                "returned no vector for the query"
            )
        return records[0].vector

    def _cap_per_document(
        self,
        matches: list,
        limit: int,
    ) -> tuple[list[RetrievedChunk], int]:
        """Take the best matches, allowing at most max_per_document per file."""

        seen: dict[str, int] = {}
        chunks: list[RetrievedChunk] = []
        dropped = 0

        for match in matches:
            if len(chunks) >= limit:
                break
            if self.max_per_document is not None:
                key = str(match.metadata.get(self.document_key, match.chunk_id))
                if seen.get(key, 0) >= self.max_per_document:
                    dropped += 1
                    continue
                seen[key] = seen.get(key, 0) + 1

            chunks.append(
                RetrievedChunk(
                    chunk_id=match.chunk_id,
                    rank=len(chunks) + 1,
                    score=match.score,
                    text=match.text,
                    metadata=match.metadata,
                )
            )

        return chunks, dropped
