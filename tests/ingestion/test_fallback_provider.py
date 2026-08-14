from __future__ import annotations

import unittest

from rag_app.ingestion.embeddings.base import EmbeddingRecord
from rag_app.ingestion.embeddings.fallback_provider import (
    FallbackEmbeddingError,
    FallbackEmbeddingProvider,
)


class FakeProvider:
    def __init__(self, provider_name: str, model_name: str, *, fails: bool = False) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.fails = fails
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[EmbeddingRecord]:
        self.calls.append(list(texts))
        if self.fails:
            raise RuntimeError(f"{self.provider_name} failed")
        return [
            EmbeddingRecord(chunk_id=str(i), vector=[0.0], model=self.model_name)
            for i in range(len(texts))
        ]


class FallbackEmbeddingProviderTests(unittest.TestCase):
    def test_empty_provider_chain_raises_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            FallbackEmbeddingProvider(providers=())

    def test_model_name_reflects_primary_provider(self) -> None:
        primary = FakeProvider("primary", "primary-model")
        secondary = FakeProvider("secondary", "secondary-model")

        chain = FallbackEmbeddingProvider(providers=(primary, secondary))

        self.assertEqual("primary-model", chain.model_name)

    def test_uses_primary_result_without_calling_secondary(self) -> None:
        primary = FakeProvider("primary", "primary-model")
        secondary = FakeProvider("secondary", "secondary-model")
        chain = FallbackEmbeddingProvider(providers=(primary, secondary))

        records = chain.embed_texts(["hello"])

        self.assertEqual(["primary-model"], [r.model for r in records])
        self.assertEqual(1, len(primary.calls))
        self.assertEqual(0, len(secondary.calls))

    def test_falls_through_to_secondary_when_primary_fails(self) -> None:
        primary = FakeProvider("primary", "primary-model", fails=True)
        secondary = FakeProvider("secondary", "secondary-model")
        chain = FallbackEmbeddingProvider(providers=(primary, secondary))

        records = chain.embed_texts(["hello"])

        self.assertEqual(["secondary-model"], [r.model for r in records])
        self.assertEqual(1, len(primary.calls))
        self.assertEqual(1, len(secondary.calls))

    def test_raises_fallback_error_when_all_providers_fail(self) -> None:
        primary = FakeProvider("primary", "primary-model", fails=True)
        secondary = FakeProvider("secondary", "secondary-model", fails=True)
        chain = FallbackEmbeddingProvider(providers=(primary, secondary))

        with self.assertRaises(FallbackEmbeddingError) as ctx:
            chain.embed_texts(["hello"])

        self.assertIn("primary/primary-model", str(ctx.exception))
        self.assertIn("secondary/secondary-model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
