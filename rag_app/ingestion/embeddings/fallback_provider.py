"""Embedding provider that falls through an ordered chain of providers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rag_app.ingestion.embeddings.base import (
    EmbeddingProvider,
    EmbeddingRecord,
    ProgressCallback,
)

logger = logging.getLogger(__name__)


class FallbackEmbeddingError(RuntimeError):
    """Raised when every provider in the fallback chain fails."""


@dataclass(slots=True)
class FallbackEmbeddingProvider:
    """Tries each provider in order, falling through on any exception."""

    providers: tuple[EmbeddingProvider, ...]
    provider_name: str = field(default="fallback-chain", init=False)

    def __post_init__(self) -> None:
        if not self.providers:
            raise ValueError("providers must contain at least one EmbeddingProvider")

    @property
    def model_name(self) -> str:
        return self.providers[0].model_name

    def embed_texts(
        self,
        texts: list[str],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> list[EmbeddingRecord]:
        failures: list[str] = []
        for provider in self.providers:
            try:
                return provider.embed_texts(texts, progress_callback=progress_callback)
            except Exception as exc:
                failures.append(f"{provider.provider_name}/{provider.model_name}: {exc}")
                logger.warning(
                    "Embedding provider %s/%s failed, trying next fallback: %s",
                    provider.provider_name,
                    provider.model_name,
                    exc,
                )

        raise FallbackEmbeddingError(
            "All embedding providers failed: " + "; ".join(failures)
        )
