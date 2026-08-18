"""Nomic Embed Text v1.5 embedding provider running locally via sentence-transformers.

The sole embedding backend for the project: runs fully offline with no API
key or rate limits, producing 768-dimension vectors. Nomic serves this exact
model over a hosted API too, so the same vectors stay reproducible if an
online deployment is added later.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import logfire

from rag_app.ingestion.embeddings.base import EmbeddingRecord, ProgressCallback

DEFAULT_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
NATIVE_DIMENSIONALITY = 768
DEFAULT_OUTPUT_DIMENSIONALITY = NATIVE_DIMENSIONALITY
DEFAULT_BATCH_SIZE = 32

# Nomic Embed is prefix-conditioned: documents and queries must be tagged with
# different task prefixes or retrieval quality degrades badly. The ingestion
# pipeline only ever embeds documents; retrieval must use DEFAULT_QUERY_PREFIX.
DEFAULT_DOCUMENT_PREFIX = "search_document: "
DEFAULT_QUERY_PREFIX = "search_query: "

LAYER_NORM_EPSILON = 1e-5


class NomicEmbeddingConfigError(RuntimeError):
    """Raised when the Nomic embedding provider is misconfigured."""


class NomicEmbeddingRuntimeError(RuntimeError):
    """Raised when local Nomic inference fails."""


@dataclass(slots=True, kw_only=True)
class NomicEmbeddingProvider:
    model_name: str = DEFAULT_MODEL_NAME
    task_prefix: str = DEFAULT_DOCUMENT_PREFIX
    output_dimensionality: int = DEFAULT_OUTPUT_DIMENSIONALITY
    batch_size: int = DEFAULT_BATCH_SIZE
    normalize: bool = True
    device: str | None = None
    cache_folder: str | None = None
    provider_name: str = field(default="nomic-local", init=False)
    # Injectable for tests; otherwise lazily loaded on first embed_texts call
    # so that constructing the provider (e.g. as a fallback that never runs)
    # does not pay the model load cost.
    _model: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise NomicEmbeddingConfigError("batch_size must be positive")
        if not 0 < self.output_dimensionality <= NATIVE_DIMENSIONALITY:
            raise NomicEmbeddingConfigError(
                "output_dimensionality must be between 1 and "
                f"{NATIVE_DIMENSIONALITY}, got {self.output_dimensionality}"
            )

        self.device = self.device or os.getenv("NOMIC_EMBED_DEVICE") or None
        self.cache_folder = self.cache_folder or os.getenv("NOMIC_EMBED_CACHE_DIR") or None

    def embed_texts(
        self,
        texts: list[str],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> list[EmbeddingRecord]:
        if not texts:
            return []

        model = self._ensure_model()
        vectors: list[list[float]] = []
        total_batches = math.ceil(len(texts) / self.batch_size)
        for batch_index, start in enumerate(range(0, len(texts), self.batch_size), start=1):
            batch = texts[start : start + self.batch_size]
            with logfire.span(
                "embed batch {batch_index}/{total_batches} ({remaining} chunks left)",
                batch_index=batch_index,
                total_batches=total_batches,
                batch_size=len(batch),
                remaining=len(texts) - start,
            ):
                vectors.extend(self._embed_batch(model, batch))
            if progress_callback is not None:
                progress_callback(len(vectors), len(texts))

        # embed_texts only receives raw strings, so the real chunk_id is not
        # known here; callers that have chunk ids (e.g. the ingestion
        # pipeline) must remap chunk_id onto the returned records by position.
        return [
            EmbeddingRecord(
                chunk_id=str(index),
                vector=vector,
                model=self.model_name,
                metadata={"task_prefix": self.task_prefix},
            )
            for index, vector in enumerate(vectors)
        ]

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise NomicEmbeddingConfigError(
                "sentence-transformers is required for NomicEmbeddingProvider; "
                "install it with `pip install sentence-transformers`."
            ) from exc

        with logfire.span(
            "load_embedding_model",
            model=self.model_name,
            device=self.device or "auto",
        ):
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=self.cache_folder,
                )
            except Exception as exc:
                raise NomicEmbeddingConfigError(
                    f"Could not load local embedding model {self.model_name}: {exc}"
                ) from exc

        return self._model

    def _embed_batch(self, model: Any, batch: list[str]) -> list[list[float]]:
        prefixed = [f"{self.task_prefix}{text}" for text in batch]
        try:
            raw_vectors = model.encode(
                prefixed,
                batch_size=len(prefixed),
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise NomicEmbeddingRuntimeError(f"Nomic embedding request failed: {exc}") from exc

        return [self._postprocess(vector) for vector in raw_vectors]

    def _postprocess(self, vector: Any) -> list[float]:
        values = [float(value) for value in vector]
        # Nomic v1.5 is a Matryoshka model: to shorten a vector, layer-norm the
        # full-width output first, then truncate, then re-normalize.
        if self.output_dimensionality < len(values):
            values = _layer_norm(values)[: self.output_dimensionality]
        return _l2_normalize(values) if self.normalize else values


def _layer_norm(values: list[float]) -> list[float]:
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    scale = math.sqrt(variance + LAYER_NORM_EPSILON)
    return [(value - mean) / scale for value in values]


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]
