"""Embedding contracts.

Concrete providers (e.g. nomic_provider.NomicEmbeddingProvider) live
alongside this module. Vector-store/indexing/retrieval integration is a
later phase.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

# Called after each completed batch with (completed_texts, total_texts) so
# callers can report live progress without owning the batching loop.
ProgressCallback = Callable[[int, int], None]


@dataclass(slots=True)
class EmbeddingRecord:
    """One embedded chunk: its vector plus the model and metadata that produced it."""

    chunk_id: str
    vector: list[float]
    model: str
    metadata: dict = field(default_factory=dict)


class EmbeddingProvider(Protocol):
    """Contract implemented by every embedding backend (local or hosted)."""

    provider_name: str
    model_name: str

    def embed_texts(
        self,
        texts: list[str],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> list[EmbeddingRecord]:
        """Embed texts, reporting per-batch progress to progress_callback."""
