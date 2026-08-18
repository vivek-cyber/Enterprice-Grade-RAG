from __future__ import annotations

import math
import unittest

from rag_app.ingestion.embeddings.nomic_provider import (
    DEFAULT_DOCUMENT_PREFIX,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIMENSIONALITY,
    NATIVE_DIMENSIONALITY,
    NomicEmbeddingConfigError,
    NomicEmbeddingProvider,
    NomicEmbeddingRuntimeError,
)


class FakeModel:
    """Stands in for a loaded SentenceTransformer."""

    def __init__(self, *, fails: bool = False, dim: int = NATIVE_DIMENSIONALITY) -> None:
        self.fails = fails
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        if self.fails:
            raise RuntimeError("inference exploded")
        # Deterministic, non-uniform vectors so normalization is observable.
        return [
            [float(len(text) + offset) for offset in range(self.dim)] for text in texts
        ]


def _make_provider(**overrides) -> tuple[NomicEmbeddingProvider, FakeModel]:
    model = overrides.pop("model", None) or FakeModel()
    provider = NomicEmbeddingProvider(_model=model, **overrides)
    return provider, model


class NomicEmbeddingProviderTests(unittest.TestCase):
    def test_defaults_are_768_dim_nomic_v15(self) -> None:
        provider, _ = _make_provider()

        self.assertEqual(DEFAULT_MODEL_NAME, provider.model_name)
        self.assertEqual(768, DEFAULT_OUTPUT_DIMENSIONALITY)
        self.assertEqual(768, provider.output_dimensionality)
        self.assertEqual("nomic-local", provider.provider_name)

    def test_non_positive_batch_size_raises_at_construction(self) -> None:
        with self.assertRaises(NomicEmbeddingConfigError):
            NomicEmbeddingProvider(_model=FakeModel(), batch_size=0)

    def test_output_dimensionality_above_native_raises_at_construction(self) -> None:
        with self.assertRaises(NomicEmbeddingConfigError):
            NomicEmbeddingProvider(_model=FakeModel(), output_dimensionality=1024)

    def test_embed_texts_empty_list_returns_empty_without_calling_model(self) -> None:
        provider, model = _make_provider()

        records = provider.embed_texts([])

        self.assertEqual([], records)
        self.assertEqual([], model.calls)

    def test_embed_texts_applies_document_task_prefix(self) -> None:
        provider, model = _make_provider()

        provider.embed_texts(["hello"])

        self.assertEqual([[f"{DEFAULT_DOCUMENT_PREFIX}hello"]], model.calls)

    def test_embed_texts_batches_requests_by_batch_size(self) -> None:
        provider, model = _make_provider(batch_size=2)

        provider.embed_texts(["a", "bb", "ccc", "dddd", "eeeee"])

        self.assertEqual([2, 2, 1], [len(call) for call in model.calls])

    def test_embed_texts_returns_768_dim_vectors(self) -> None:
        provider, _ = _make_provider()

        records = provider.embed_texts(["hello", "world"])

        self.assertEqual(2, len(records))
        for record in records:
            self.assertEqual(768, len(record.vector))

    def test_vectors_are_l2_normalized_by_default(self) -> None:
        provider, _ = _make_provider()

        records = provider.embed_texts(["hello"])

        norm = math.sqrt(sum(value * value for value in records[0].vector))
        self.assertAlmostEqual(1.0, norm, places=6)

    def test_normalization_can_be_disabled(self) -> None:
        provider, _ = _make_provider(normalize=False)

        records = provider.embed_texts(["hello"])

        norm = math.sqrt(sum(value * value for value in records[0].vector))
        self.assertNotAlmostEqual(1.0, norm, places=6)

    def test_matryoshka_truncation_yields_normalized_shorter_vectors(self) -> None:
        provider, _ = _make_provider(output_dimensionality=256)

        records = provider.embed_texts(["hello"])

        self.assertEqual(256, len(records[0].vector))
        norm = math.sqrt(sum(value * value for value in records[0].vector))
        self.assertAlmostEqual(1.0, norm, places=6)

    def test_embed_texts_uses_positional_placeholder_chunk_ids(self) -> None:
        provider, _ = _make_provider(batch_size=2)

        records = provider.embed_texts(["a", "bb", "ccc"])

        self.assertEqual(["0", "1", "2"], [record.chunk_id for record in records])

    def test_records_carry_model_name_and_task_prefix_metadata(self) -> None:
        provider, _ = _make_provider()

        records = provider.embed_texts(["hello"])

        self.assertEqual(DEFAULT_MODEL_NAME, records[0].model)
        self.assertEqual({"task_prefix": DEFAULT_DOCUMENT_PREFIX}, records[0].metadata)

    def test_inference_failure_raises_runtime_error(self) -> None:
        provider, _ = _make_provider(model=FakeModel(fails=True))

        with self.assertRaises(NomicEmbeddingRuntimeError):
            provider.embed_texts(["hello"])

    def test_progress_callback_reports_cumulative_counts_per_batch(self) -> None:
        provider, _ = _make_provider(batch_size=2)
        seen: list[tuple[int, int]] = []

        provider.embed_texts(
            ["a", "bb", "ccc", "dddd", "eeeee"],
            progress_callback=lambda done, total: seen.append((done, total)),
        )

        self.assertEqual([(2, 5), (4, 5), (5, 5)], seen)

    def test_order_is_preserved_across_batches(self) -> None:
        provider, _ = _make_provider(batch_size=2, normalize=False)

        records = provider.embed_texts(["a", "bb", "ccc", "dddd", "eeeee"])

        # FakeModel encodes the prefixed text's length into the first component,
        # so values stay strictly increasing with the original input lengths.
        prefix_length = len(DEFAULT_DOCUMENT_PREFIX)
        self.assertEqual(
            [float(prefix_length + n) for n in (1, 2, 3, 4, 5)],
            [record.vector[0] for record in records],
        )


if __name__ == "__main__":
    unittest.main()
