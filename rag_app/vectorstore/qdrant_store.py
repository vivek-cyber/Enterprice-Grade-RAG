"""Qdrant vector store implementation."""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass, field

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from rag_app.vectorstore.base import MetadataFilter, VectorMatch, VectorPoint

DEFAULT_DISTANCE = qdrant_models.Distance.COSINE
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_UPSERT_BATCH_SIZE = 256

# Payload fields ensure_collection() indexes for filtered search. Qdrant
# Cloud rejects an unindexed filter outright (400 "Index required but not
# found") rather than falling back to a slow scan, so any field callers
# filter on via MetadataFilter must be listed here.
DEFAULT_INDEXED_FIELDS: tuple[str, ...] = (
    "source_type",
    "source_name",
    "source_extension",
)

# Chunk ids are sha256 hex digests, which aren't valid Qdrant point ids
# (unsigned int or UUID only). This namespace derives a stable UUID per
# chunk_id so re-ingesting the same chunk upserts the same point.
_POINT_ID_NAMESPACE = uuid.UUID("5f1d9f0e-9b7a-4b7b-8f2f-9f0a5b6c7d8e")


class QdrantConfigError(RuntimeError):
    """Raised when the Qdrant vector store is misconfigured."""


@dataclass(slots=True, kw_only=True)
class QdrantVectorStore:
    """Vector store backed by Qdrant (Cloud or self-hosted)."""

    collection_name: str
    url: str | None = None
    api_key: str | None = None
    distance: qdrant_models.Distance = DEFAULT_DISTANCE
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE
    indexed_fields: tuple[str, ...] = DEFAULT_INDEXED_FIELDS
    store_name: str = field(default="qdrant", init=False)
    _client: QdrantClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve connection settings from args or environment and open the client."""

        resolved_url = self.url or os.getenv("QDRANT_URL")
        if not resolved_url:
            raise QdrantConfigError("QDRANT_URL environment variable is not set.")

        self.url = resolved_url
        self.api_key = self.api_key or os.getenv("QDRANT_API_KEY")
        self._client = QdrantClient(
            url=self.url, api_key=self.api_key, timeout=self.timeout
        )

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection with the given vector size if it doesn't exist yet.

        Also creates a keyword payload index on each of indexed_fields. Qdrant
        Cloud refuses to filter on an unindexed field at all (see
        _to_qdrant_filter), so a collection created without this step cannot
        serve a filtered search until indexes are added after the fact.
        """

        if self._client.collection_exists(self.collection_name):
            return
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=vector_size,
                distance=self.distance,
            ),
        )
        for field_name in self.indexed_fields:
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )

    def upsert(self, points: list[VectorPoint]) -> None:
        """Insert or overwrite points in batches, keyed by a deterministic point id."""

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

    def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        filters: MetadataFilter | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorMatch]:
        """Return the nearest points to vector, most similar first."""

        response = self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            query_filter=_to_qdrant_filter(filters),
            score_threshold=score_threshold,
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
        """Remove points by chunk_id."""

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


def _to_qdrant_filter(filters: MetadataFilter | None) -> qdrant_models.Filter | None:
    """Translate a MetadataFilter into Qdrant's own filter model.

    Returns None for an absent or empty filter so the caller sends no filter
    at all rather than an empty one, which Qdrant would still evaluate.

    Filtered search does not require a payload index, but Qdrant can only take
    its fast path when one exists. At this corpus size the difference is
    negligible; add payload indexes on the fields you filter by before the
    collection grows.
    """

    if filters is None or filters.is_empty():
        return None

    conditions: list[qdrant_models.FieldCondition] = [
        qdrant_models.FieldCondition(
            key=key,
            match=qdrant_models.MatchValue(value=value),
        )
        for key, value in filters.equals.items()
    ]
    conditions.extend(
        qdrant_models.FieldCondition(
            key=key,
            match=qdrant_models.MatchAny(any=list(values)),
        )
        for key, values in filters.any_of.items()
        if values
    )
    return qdrant_models.Filter(must=conditions) if conditions else None
