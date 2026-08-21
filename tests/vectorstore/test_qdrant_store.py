"""Tests for QdrantVectorStore: config resolution, batching, search, and delete."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_app.vectorstore.base import VectorPoint
from rag_app.vectorstore.qdrant_store import (
    QdrantConfigError,
    QdrantVectorStore,
    _point_id,
)


def _make_store(**overrides) -> tuple[QdrantVectorStore, MagicMock]:
    fake_client = MagicMock()
    with patch(
        "rag_app.vectorstore.qdrant_store.QdrantClient",
        return_value=fake_client,
    ):
        store = QdrantVectorStore(
            collection_name="chunks",
            url=overrides.pop("url", "https://example.qdrant.io"),
            api_key=overrides.pop("api_key", "test-key"),
            **overrides,
        )
    return store, fake_client


class QdrantVectorStoreTests(unittest.TestCase):
    """Verifies config resolution, upsert/search/delete mapping, and point-id stability."""

    def test_missing_url_raises_at_construction(self) -> None:
        """Missing url raises at construction."""

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(QdrantConfigError):
                QdrantVectorStore(collection_name="chunks")

    def test_url_from_env_used_when_not_passed_explicitly(self) -> None:
        """Url from env used when not passed explicitly."""

        with (
            patch.dict(
                "os.environ", {"QDRANT_URL": "https://env.qdrant.io"}, clear=True
            ),
            patch(
                "rag_app.vectorstore.qdrant_store.QdrantClient",
                return_value=MagicMock(),
            ),
        ):
            store = QdrantVectorStore(collection_name="chunks")

        self.assertEqual("https://env.qdrant.io", store.url)

    def test_ensure_collection_skips_creation_when_collection_exists(self) -> None:
        """Ensure collection skips creation when collection exists."""

        store, fake_client = _make_store()
        fake_client.collection_exists.return_value = True

        store.ensure_collection(3)

        fake_client.create_collection.assert_not_called()

    def test_ensure_collection_creates_collection_when_missing(self) -> None:
        """Ensure collection creates collection when missing."""

        store, fake_client = _make_store()
        fake_client.collection_exists.return_value = False

        store.ensure_collection(3)

        fake_client.create_collection.assert_called_once()
        _, kwargs = fake_client.create_collection.call_args
        self.assertEqual("chunks", kwargs["collection_name"])
        self.assertEqual(3, kwargs["vectors_config"].size)

    def test_upsert_maps_chunk_ids_to_stable_point_ids(self) -> None:
        """Upsert maps chunk ids to stable point ids."""

        store, fake_client = _make_store()
        points = [
            VectorPoint(
                chunk_id="abc", vector=[0.1, 0.2], text="hello", metadata={"k": "v"}
            ),
        ]

        store.upsert(points)

        fake_client.upsert.assert_called_once()
        _, kwargs = fake_client.upsert.call_args
        self.assertEqual("chunks", kwargs["collection_name"])
        sent_point = kwargs["points"][0]
        self.assertEqual(_point_id("abc"), sent_point.id)
        self.assertEqual([0.1, 0.2], sent_point.vector)
        self.assertEqual(
            {"chunk_id": "abc", "text": "hello", "k": "v"},
            sent_point.payload,
        )

    def test_upsert_splits_points_into_multiple_batches(self) -> None:
        """Upsert splits points into multiple batches."""

        store, fake_client = _make_store(upsert_batch_size=2)
        points = [
            VectorPoint(chunk_id=str(i), vector=[float(i)], text=str(i), metadata={})
            for i in range(5)
        ]

        store.upsert(points)

        self.assertEqual(3, fake_client.upsert.call_count)
        batch_sizes = [
            len(call.kwargs["points"]) for call in fake_client.upsert.call_args_list
        ]
        self.assertEqual([2, 2, 1], batch_sizes)

    def test_upsert_with_no_points_does_not_call_client(self) -> None:
        """Upsert with no points does not call client."""

        store, fake_client = _make_store()

        store.upsert([])

        fake_client.upsert.assert_not_called()

    def test_search_maps_results_to_vector_matches(self) -> None:
        """Search maps results to vector matches."""

        store, fake_client = _make_store()
        fake_client.query_points.return_value = SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=_point_id("abc"),
                    score=0.9,
                    payload={"chunk_id": "abc", "text": "hello", "source": "doc-1"},
                )
            ]
        )

        results = store.search([0.1, 0.2], limit=5)

        self.assertEqual(1, len(results))
        self.assertEqual("abc", results[0].chunk_id)
        self.assertEqual(0.9, results[0].score)
        self.assertEqual("hello", results[0].text)
        self.assertEqual({"source": "doc-1"}, results[0].metadata)

    def test_delete_maps_chunk_ids_to_stable_point_ids(self) -> None:
        """Delete maps chunk ids to stable point ids."""

        store, fake_client = _make_store()

        store.delete(["abc", "def"])

        fake_client.delete.assert_called_once()
        _, kwargs = fake_client.delete.call_args
        self.assertEqual("chunks", kwargs["collection_name"])
        self.assertEqual(
            [_point_id("abc"), _point_id("def")],
            kwargs["points_selector"].points,
        )

    def test_delete_with_no_ids_does_not_call_client(self) -> None:
        """Delete with no ids does not call client."""

        store, fake_client = _make_store()

        store.delete([])

        fake_client.delete.assert_not_called()

    def test_point_id_is_deterministic_per_chunk_id(self) -> None:
        """Point id is deterministic per chunk id."""

        self.assertEqual(_point_id("abc"), _point_id("abc"))
        self.assertNotEqual(_point_id("abc"), _point_id("def"))


if __name__ == "__main__":
    unittest.main()
