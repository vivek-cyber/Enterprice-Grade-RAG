"""Tests for DenseRetriever: query embedding, oversampling, and per-doc capping."""

from __future__ import annotations

import unittest

from rag_app.ingestion.embeddings.base import EmbeddingRecord, ProgressCallback
from rag_app.ingestion.embeddings.nomic_provider import (
    DEFAULT_DOCUMENT_PREFIX,
    DEFAULT_QUERY_PREFIX,
)
from rag_app.retrieval.base import RetrievalResult
from rag_app.retrieval.dense import (
    DenseRetriever,
    QueryPrefixError,
    build_query_embedder,
)
from rag_app.vectorstore.base import MetadataFilter, VectorMatch


class FakeEmbedder:
    """Minimal EmbeddingProvider stand-in with a fixed, inspectable vector."""

    def __init__(self, *, task_prefix: str = DEFAULT_QUERY_PREFIX) -> None:
        """Record the configured prefix and set up empty call tracking."""

        self.provider_name = "fake-embedder"
        self.model_name = "fake-model"
        self.task_prefix = task_prefix
        self.calls: list[list[str]] = []

    def embed_texts(
        self,
        texts: list[str],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> list[EmbeddingRecord]:
        """Record the call and return one fixed vector per input text."""

        self.calls.append(list(texts))
        return [
            EmbeddingRecord(chunk_id=str(i), vector=[1.0, 0.0], model=self.model_name)
            for i in range(len(texts))
        ]


class FakeStore:
    """Minimal VectorStore stand-in that returns a canned match list."""

    def __init__(self, matches: list[VectorMatch]) -> None:
        """Store canned matches and record every search() call's arguments."""

        self.store_name = "fake-store"
        self._matches = matches
        self.search_calls: list[dict] = []

    def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        filters: MetadataFilter | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorMatch]:
        """Record the call's arguments and return the canned matches, truncated."""

        self.search_calls.append(
            {
                "vector": vector,
                "limit": limit,
                "filters": filters,
                "score_threshold": score_threshold,
            }
        )
        return self._matches[:limit]


def _match(chunk_id: str, score: float, *, source: str = "doc.pdf") -> VectorMatch:
    return VectorMatch(
        chunk_id=chunk_id,
        score=score,
        text=f"text for {chunk_id}",
        metadata={"source_path": source, "source_name": source},
    )


class DenseRetrieverConstructionTests(unittest.TestCase):
    def test_rejects_document_prefixed_embedder(self) -> None:
        embedder = FakeEmbedder(task_prefix=DEFAULT_DOCUMENT_PREFIX)
        with self.assertRaises(QueryPrefixError):
            DenseRetriever(embedding_provider=embedder, vector_store=FakeStore([]))

    def test_accepts_query_prefixed_embedder(self) -> None:
        embedder = FakeEmbedder(task_prefix=DEFAULT_QUERY_PREFIX)
        DenseRetriever(embedding_provider=embedder, vector_store=FakeStore([]))

    def test_rejects_oversample_factor_below_one(self) -> None:
        embedder = FakeEmbedder()
        with self.assertRaises(ValueError):
            DenseRetriever(
                embedding_provider=embedder,
                vector_store=FakeStore([]),
                oversample_factor=0,
            )

    def test_rejects_max_per_document_below_one(self) -> None:
        embedder = FakeEmbedder()
        with self.assertRaises(ValueError):
            DenseRetriever(
                embedding_provider=embedder,
                vector_store=FakeStore([]),
                max_per_document=0,
            )

    def test_build_query_embedder_uses_query_prefix(self) -> None:
        embedder = build_query_embedder()
        self.assertEqual(DEFAULT_QUERY_PREFIX, embedder.task_prefix)


class DenseRetrieverRetrieveTests(unittest.TestCase):
    def test_embeds_query_with_stripped_text(self) -> None:
        embedder = FakeEmbedder()
        retriever = DenseRetriever(
            embedding_provider=embedder, vector_store=FakeStore([])
        )

        retriever.retrieve("  hello world  ")

        self.assertEqual([["hello world"]], embedder.calls)

    def test_rejects_empty_query(self) -> None:
        retriever = DenseRetriever(
            embedding_provider=FakeEmbedder(), vector_store=FakeStore([])
        )
        with self.assertRaises(ValueError):
            retriever.retrieve("   ")

    def test_rejects_limit_below_one(self) -> None:
        retriever = DenseRetriever(
            embedding_provider=FakeEmbedder(), vector_store=FakeStore([])
        )
        with self.assertRaises(ValueError):
            retriever.retrieve("query", limit=0)

    def test_oversamples_store_search_by_configured_factor(self) -> None:
        embedder = FakeEmbedder()
        store = FakeStore([])
        retriever = DenseRetriever(
            embedding_provider=embedder, vector_store=store, oversample_factor=4
        )

        retriever.retrieve("query", limit=10)

        self.assertEqual(40, store.search_calls[0]["limit"])

    def test_passes_filters_and_score_threshold_through_to_store(self) -> None:
        store = FakeStore([])
        retriever = DenseRetriever(
            embedding_provider=FakeEmbedder(),
            vector_store=store,
            score_threshold=0.5,
        )
        filters = MetadataFilter(equals={"source_type": ".pdf"})

        retriever.retrieve("query", filters=filters)

        call = store.search_calls[0]
        self.assertIs(filters, call["filters"])
        self.assertEqual(0.5, call["score_threshold"])

    def test_returns_matches_ranked_in_store_order(self) -> None:
        matches = [_match("a", 0.9), _match("b", 0.8), _match("c", 0.7)]
        retriever = DenseRetriever(
            embedding_provider=FakeEmbedder(), vector_store=FakeStore(matches)
        )

        result = retriever.retrieve("query", limit=3)

        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual(["a", "b", "c"], [c.chunk_id for c in result.chunks])
        self.assertEqual([1, 2, 3], [c.rank for c in result.chunks])

    def test_caps_matches_per_document_and_reports_dropped_count(self) -> None:
        matches = [
            _match("a1", 0.9, source="doc-a.pdf"),
            _match("a2", 0.85, source="doc-a.pdf"),
            _match("a3", 0.8, source="doc-a.pdf"),
            _match("a4", 0.75, source="doc-a.pdf"),
            _match("b1", 0.7, source="doc-b.pdf"),
        ]
        retriever = DenseRetriever(
            embedding_provider=FakeEmbedder(),
            vector_store=FakeStore(matches),
            max_per_document=2,
        )

        result = retriever.retrieve("query", limit=10)

        self.assertEqual(["a1", "a2", "b1"], [c.chunk_id for c in result.chunks])
        self.assertEqual(2, result.dropped_by_document_cap)
        self.assertEqual(5, result.candidates_considered)

    def test_max_per_document_none_disables_capping(self) -> None:
        matches = [
            _match(f"c{i}", 1.0 - i * 0.01, source="doc-a.pdf") for i in range(5)
        ]
        retriever = DenseRetriever(
            embedding_provider=FakeEmbedder(),
            vector_store=FakeStore(matches),
            max_per_document=None,
        )

        result = retriever.retrieve("query", limit=5)

        self.assertEqual(5, len(result.chunks))
        self.assertEqual(0, result.dropped_by_document_cap)

    def test_stops_at_limit_even_with_more_candidates_available(self) -> None:
        matches = [
            _match(f"c{i}", 1.0 - i * 0.01, source=f"doc-{i}.pdf") for i in range(10)
        ]
        retriever = DenseRetriever(
            embedding_provider=FakeEmbedder(), vector_store=FakeStore(matches)
        )

        result = retriever.retrieve("query", limit=3)

        self.assertEqual(3, len(result.chunks))

    def test_empty_store_response_yields_empty_result(self) -> None:
        retriever = DenseRetriever(
            embedding_provider=FakeEmbedder(), vector_store=FakeStore([])
        )

        result = retriever.retrieve("query")

        self.assertTrue(result.is_empty)
        self.assertEqual(0, len(result))


if __name__ == "__main__":
    unittest.main()
