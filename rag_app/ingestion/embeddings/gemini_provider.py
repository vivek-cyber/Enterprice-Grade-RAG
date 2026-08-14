"""Gemini Developer API embedding provider."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from rag_app.ingestion.embeddings.base import EmbeddingRecord

DEFAULT_MODEL_NAME = "gemini-embedding-2-preview"
DEFAULT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_BACKOFF_SECONDS = 1.0
RATE_LIMIT_STATUS_CODE = 429


class GeminiEmbeddingConfigError(RuntimeError):
    """Raised when the Gemini embedding provider is misconfigured."""


class GeminiEmbeddingAPIError(RuntimeError):
    """Raised when the Gemini embeddings API fails after retries are exhausted."""


@dataclass(slots=True, kw_only=True)
class GeminiEmbeddingProvider:
    model_name: str = DEFAULT_MODEL_NAME
    task_type: str = DEFAULT_TASK_TYPE
    output_dimensionality: int | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    max_retries: int = DEFAULT_MAX_RETRIES
    initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS
    api_key: str | None = None
    provider_name: str = field(default="gemini", init=False)
    _client: genai.Client = field(init=False, repr=False)
    _sleep: Callable[[float], None] = field(default=time.sleep, repr=False)

    def __post_init__(self) -> None:
        resolved_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise GeminiEmbeddingConfigError(
                "GEMINI_API_KEY environment variable is not set."
            )
        if self.batch_size <= 0:
            raise GeminiEmbeddingConfigError("batch_size must be positive")

        self.api_key = resolved_key
        self._client = genai.Client(api_key=resolved_key)

    def embed_texts(self, texts: list[str]) -> list[EmbeddingRecord]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))

        # embed_texts only receives raw strings, so the real chunk_id is not
        # known here; callers that have chunk ids (e.g. the ingestion
        # pipeline) must remap chunk_id onto the returned records by position.
        return [
            EmbeddingRecord(
                chunk_id=str(index),
                vector=vector,
                model=self.model_name,
                metadata={"task_type": self.task_type},
            )
            for index, vector in enumerate(vectors)
        ]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        attempt = 0
        backoff = self.initial_backoff_seconds
        while True:
            try:
                response = self._client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                    config=genai_types.EmbedContentConfig(
                        task_type=self.task_type,
                        output_dimensionality=self.output_dimensionality,
                    ),
                )
                return [embedding.values for embedding in response.embeddings]
            except genai_errors.APIError as exc:
                attempt += 1
                if exc.code != RATE_LIMIT_STATUS_CODE or attempt > self.max_retries:
                    raise GeminiEmbeddingAPIError(
                        f"Gemini embedding request failed: {exc}"
                    ) from exc
                self._sleep(backoff)
                backoff *= 2
