# `rag_app.vectorstore`

Vector database contracts and implementations, used by the ingestion
pipeline to upsert embedded chunks and (in a future retrieval phase) to
search them.

- **`base.py`** — the `VectorStore` Protocol
  (`ensure_collection`/`upsert`/`search`/`delete`) plus the two data
  contracts every store speaks: `VectorPoint` (one embedded chunk going in)
  and `VectorMatch` (one search result coming out). Callers should depend on
  this Protocol, not on `QdrantVectorStore` directly, so the backing store
  can be swapped without touching ingestion or retrieval code.
- **`qdrant_store.py`** — the only implementation today. Talks to
  [Qdrant](https://qdrant.tech/) (Cloud or self-hosted) via `qdrant-client`.
  Notable details:
  - Chunk IDs are sha256 hex digests, which aren't valid Qdrant point IDs
    (Qdrant requires an unsigned int or a UUID). `_point_id()` derives a
    stable UUIDv5 from each chunk ID (via a fixed namespace UUID) so
    re-ingesting the same chunk always upserts the same point instead of
    duplicating it.
  - `upsert()` batches points (`upsert_batch_size`, default 256) so a single
    call doesn't try to send an entire corpus in one request.
  - Configuration (`QDRANT_URL` / `QDRANT_API_KEY`) can be passed explicitly
    or picked up from the environment; see the root `.env.example`.

## Adding a new store

Implement the four `VectorStore` methods on a new class (see
`qdrant_store.py` for the reference shape — a `dataclass(slots=True,
kw_only=True)` with a `store_name` and lazy client setup in
`__post_init__`), then pass an instance of it to
`rag_app.ingestion.pipeline.ingest_folder(vector_store=...)`.
