"""Qdrant vector store implementation."""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass, field

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from rag_app.vectorstore.base import VectorMatch, VectorPoint

DEFAULT_DISTANCE = qdrant_models.Distance.COSINE
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_UPSERT_BATCH_SIZE = 256

# Chunk ids are sha256 hex digests, which aren't valid Qdrant point ids
# (unsigned int or UUID only). This namespace derives a stable UUID per
# chunk_id so re-ingesting the same chunk upserts the same point.
_POINT_ID_NAMESPACE = uuid.UUID("5f1d9f0e-9b7a-4b7b-8f2f-9f0a5b6c7d8e")


class QdrantConfigError(RuntimeError):
    """Raised when the Qdrant vector store is misconfigured."""


@dataclass(slots=True, kw_only=True)
class QdrantVectorStore:
    collection_name: str
    url: str | None = None
    api_key: str | None = None
    distance: qdrant_models.Distance = DEFAULT_DISTANCE
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE
    store_name: str = field(default="qdrant", init=False)
    _client: QdrantClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        resolved_url = self.url or os.getenv("QDRANT_URL")
        if not resolved_url:
            raise QdrantConfigError("QDRANT_URL environment variable is not set.")

        self.url = resolved_url
        self.api_key = self.api_key or os.getenv("QDRANT_API_KEY")
        self._client = QdrantClient(url=self.url, api_key=self.api_key, timeout=self.timeout)

    def ensure_collection(self, vector_size: int) -> None:
        if self._client.collection_exists(self.collection_name):
            return
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=vector_size,
                distance=self.distance,
            ),
        )

    def upsert(self, points: list[VectorPoint]) -> None:
        if not points:
            return
        total_batches = math.ceil(len(points) / self.upsert_batch_size)
        for batch_index, start in enumerate(
            range(0, len(points), self.upsert_batch_size), start=1
        ):
            batch = points[start : start + self.upsert_batch_size]
            with logfire.span(
                "upsert batch {batch_index}/{total_batches} ({remaining} points left)",
                batch_index=batch_index,
                total_batches=total_batches,
                point_count=len(batch),
                remaining=len(points) - start,
            ):
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        qdrant_models.PointStruct(
                            id=_point_id(point.chunk_id),
                            vector=point.vector,
                            payload={
                                "chunk_id": point.chunk_id,
                                "text": point.text,
                                **point.metadata,
                            },
                        )
                        for point in batch
                    ],
                )

    def search(self, vector: list[float], *, limit: int = 10) -> list[VectorMatch]:
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
        )
        return [
            VectorMatch(
                chunk_id=result.payload.get("chunk_id", str(result.id)),
                score=result.score,
                text=result.payload.get("text", ""),
                metadata={
                    key: value
                    for key, value in result.payload.items()
                    if key not in ("chunk_id", "text")
                },
            )
            for result in response.points
        ]

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.PointIdsList(
                points=[_point_id(chunk_id) for chunk_id in chunk_ids]
            ),
        )


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))
