# `rag_app.retrieval`

Turns a user query into ranked chunks from the vector store built by
`rag_app.ingestion`. Symmetric to that package: ingestion embeds and stores
documents, retrieval embeds a query and searches for them.

- **`base.py`** — the `Retriever` Protocol (`retrieve(query, limit, filters)`)
  plus the data contracts every retriever speaks: `RetrievedChunk` (one
  ranked result, with `rank`/`score`/`text`/`metadata` and `source_name` /
  `source_path` convenience properties) and `RetrievalResult` (the full
  response, including `candidates_considered` and `dropped_by_document_cap`
  so a caller can tell *why* fewer chunks came back than asked for, without
  re-running the query). Callers should depend on this Protocol, not on
  `DenseRetriever` directly, so hybrid or reranking retrievers can be swapped
  in later without touching call sites.
- **`dense.py`** — `DenseRetriever`, the only implementation today. Embeds the
  query and searches `rag_app.vectorstore` for nearest neighbors. Notable
  details:
  - **Query/document prefix mismatch is a construction-time error, not a
    silent quality bug.** Nomic Embed is prefix-conditioned (see
    `rag_app.ingestion.embeddings.nomic_provider`): the ingestion pipeline
    embeds chunks with `search_document: `, so a query embedded with that
    same prefix lands in the wrong part of the vector space with no error
    anywhere -- it just ranks badly. `DenseRetriever.__post_init__` checks
    for this and raises `QueryPrefixError` immediately. Always build the
    query embedder with `build_query_embedder()` rather than
    `NomicEmbeddingProvider()` directly.
  - **Oversampling + per-document capping.** The store is asked for
    `limit * oversample_factor` candidates (default factor: 4), then capped
    to at most `max_per_document` chunks per source file (default: 3,
    keyed by the `source_path` payload field) before truncating to `limit`.
    Without this, one long PDF whose sections all score well can fill every
    result slot and crowd out every other document. Set
    `max_per_document=None` to disable it.
  - `score_threshold` (constructor) drops matches below a similarity cutoff
    -- returning fewer results, or none, rather than padding with weak ones.

## Filtering

`rag_app.vectorstore.base.MetadataFilter` (`equals` / `any_of`, AND-combined)
restricts which points are eligible, e.g. `source_type=".pdf"`. Qdrant Cloud
requires a payload index to filter on a field at all -- it returns a 400
rather than falling back to a slow scan -- so `QdrantVectorStore.ensure_collection()`
creates a keyword index on every field in `indexed_fields`
(`DEFAULT_INDEXED_FIELDS`: `source_type`, `source_name`, `source_extension`)
whenever it creates a new collection. Filtering on a field outside that list
needs an index added first (`QdrantClient.create_payload_index`) before it
will work against an existing collection.

## Adding a new retriever

Implement `retrieve()` on a new class (see `dense.py` for the reference
shape -- a `dataclass(slots=True, kw_only=True)` with a `retriever_name` and
validation in `__post_init__`), returning a `RetrievalResult`. A hybrid
retriever combining dense and keyword search, or a reranking wrapper around
another `Retriever`, both fit this same Protocol without callers changing.

## CLI

`scripts/query.py` is a thin wrapper: build a `DenseRetriever`, call
`retrieve()`, print the result. Use it to eyeball retrieval quality --
`python scripts/query.py "your question" --limit 5`. See its module
docstring for the full flag list (`--min-score`, `--source-type`,
`--source-name`, `--max-per-document`, `--json`).
