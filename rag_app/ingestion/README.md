# `rag_app.ingestion`

Everything needed to turn a folder of source documents into embedded,
checkpointed chunks. The entrypoint is `ingest_folder()` in `pipeline.py`;
everything else in this package is a stage it calls, in this order:

1. **`file_discovery.py`** — walk the source folder, hash each file
   (sha256), and classify it as supported or skipped based on extension.
2. **`parser/`** — one `BaseParser` subclass per file type
   (`txt_parser.py`, `html_parser.py`, `pdf_parser.py`, `docx_parser.py`,
   `pptx_parser.py`). `parser/__init__.py` maps extensions to parser
   instances via `build_parser_registry()`.
3. **`cleaners/cleaners.py`** — `clean_text()` normalizes unicode, strips
   control characters, and collapses excess whitespace/blank lines.
4. **`chunking/chunking.py`** — `chunk_document()` splits cleaned text into
   overlapping, deterministically-ID'd chunks (same input always produces
   the same chunk IDs, which is what makes checkpointing and re-ingestion
   safe).
5. **`checkpoint.py`** — `ChunkCheckpointStore` persists each file's
   parsed+chunked result to `.ingest_cache/` as soon as it's produced, keyed
   by content hash + chunk parameters, so a crashed run resumes without
   reparsing finished files.
6. **`embeddings/`** — `EmbeddingProvider` implementations that turn chunk
   text into vectors (see below).

`models.py` holds the shared dataclasses (`Document`, `DocumentChunk`,
`FileRecord`, `ParseResult`, `IngestionReport`) that every stage passes
between each other.

## Parsers (`parser/`)

Every rich-format parser (`pdf`, `docx`, `pptx`, `html`) follows the same
two-tier pattern, implemented via `parse_with_docling()` /
`parse_with_fallback()`:

- **Primary path**: convert via [Docling](https://github.com/docling-project/docling)
  (`docling_utils.py`), which does layout-aware extraction (tables, reading
  order) and exports Markdown.
- **Fallback path**: if Docling isn't installed, fails, or (for PDFs)
  silently drops pages — `docling_utils._reject_incomplete_conversion`
  treats a partial-success conversion as a failure rather than returning a
  truncated document — fall back to a format-specific plain extractor
  (`pypdf`, `python-docx`, `python-pptx`, or a raw XML/BeautifulSoup parse).

`TxtParser` has no Docling path; it just tries a list of encodings in order
(`encodings` class attribute) and records which one it used.

Add a new file type by subclassing `BaseParser`, setting `parser_name` and
`supported_extensions`, and registering the instance in
`parser/__init__.py`'s `DEFAULT_PARSERS`.

## Embeddings (`embeddings/`)

- **`base.py`** — the `EmbeddingProvider` Protocol (`embed_texts(texts,
  progress_callback=...) -> list[EmbeddingRecord]`) that every provider
  implements.
- **`nomic_provider.py`** — the only production backend: runs
  `nomic-ai/nomic-embed-text-v1.5` locally via `sentence-transformers`
  (offline, 768-dim, no rate limits). Handles Matryoshka truncation
  (`output_dimensionality`) and prefix-conditioning — **documents and
  queries need different prefixes** (`search_document: ` vs.
  `search_query: `); ingestion always uses the document prefix, so any
  future retrieval code must apply `DEFAULT_QUERY_PREFIX` itself.
- **`fallback_provider.py`** — `FallbackEmbeddingProvider` chains several
  providers and falls through to the next on any exception; not currently
  wired into `scripts/ingest.py` (which uses `NomicEmbeddingProvider`
  directly) but available for a future multi-backend setup.

## Checkpointing (`checkpoint.py`)

`ChunkCheckpointStore` is content-addressed: the cache key is a hash of
`(schema version, resolved file path, file sha256, chunk_size,
chunk_overlap)`. Editing a source file, moving it, or changing chunk
parameters naturally misses the cache instead of serving stale chunks.
Entries are written atomically (temp file + `os.replace`) so a kill mid-write
can't corrupt an entry — worst case it's just a cache miss on the next run.
