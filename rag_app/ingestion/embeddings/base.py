"""Embedding contracts.

This phase intentionally defines interfaces only. Provider implementations and
model calls belong to the retrieval/vector-indexing phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class EmbeddingRecord:
    chunk_id: str
    vector: list[float]
    model: str
    metadata: dict = field(default_factory=dict)


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str

    def embed_texts(self, texts: list[str]) -> list[EmbeddingRecord]:
        """Embed texts in a future phase."""
