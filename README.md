# Enterprise RAG (self-start)

A local-first Retrieval-Augmented Generation (RAG) pipeline: it ingests documents
(PDF, DOCX, PPTX, HTML, TXT) from a folder, parses and cleans them, splits them
into chunks, embeds the chunks with a locally-run model, and stores the vectors
in Qdrant for later retrieval. Everything except the vector store runs offline —
no API keys or per-request costs for embedding.

## Status

Only the **ingestion** stage is built and tested (see `rag_app/ingestion/`,
`rag_app/vectorstore/`, `scripts/ingest.py`). Retrieval/query and the
LangGraph-based agent workflow (see `requirements.txt`) are not implemented
yet — `langgraph` and `streamlit` are placeholder dependencies for that
future phase.

## How it fits together

```
scripts/ingest.py            CLI entrypoint
        |
        v
rag_app/ingestion/pipeline.py   ingest_folder(): orchestrates every step below
        |
        +-- file_discovery.py     walk a folder, hash + classify files
        +-- parser/                one parser class per file type (txt/html/pdf/docx/pptx)
        |     `-- docling_utils.py   shared Docling conversion helper (primary path)
        +-- cleaners/cleaners.py   normalize whitespace/unicode/control chars
        +-- chunking/chunking.py  deterministic, content-hashed chunking
        +-- checkpoint.py         per-file cache so a crash doesn't reparse finished files
        +-- embeddings/            EmbeddingProvider implementations (Nomic, fallback chain)
        `-- ../vectorstore/         VectorStore implementations (Qdrant)
```

Each stage is a small, swappable piece behind a `Protocol`/ABC
(`parser.base.BaseParser`, `embeddings.base.EmbeddingProvider`,
`vectorstore.base.VectorStore`), so new file types, embedding backends, or
vector databases can be added without touching the pipeline orchestration in
`pipeline.py`. See `rag_app/ingestion/README.md` and
`rag_app/vectorstore/README.md` for details on each layer.

## Setup

Requires Python 3.11+ (uses `X | Y` union syntax and `dataclass(slots=True)`).

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the values you need:

```bash
cp .env.example .env
```

- `QDRANT_URL` / `QDRANT_API_KEY` — required only when actually writing to
  Qdrant (not needed for `--dry-run`). Get a free cluster at
  [cloud.qdrant.io](https://cloud.qdrant.io).
- `LOGFIRE_TOKEN` — optional; ingestion runs and logs locally without it.
- `NOMIC_EMBED_DEVICE` / `NOMIC_EMBED_CACHE_DIR` — optional overrides for the
  local embedding model (device defaults to whatever `torch` auto-selects).
- `DOCLING_ACCELERATOR_DEVICE` — CPU by default. Docling's layout/table models
  exhausted GPU VRAM on large PDFs during testing, so they stay on CPU unless
  you override this and know you have headroom.

## Running ingestion

```bash
# Parse + chunk only, no embedding or vector store writes:
python scripts/ingest.py DATA/true_data --dry-run

# Full run: parse, chunk, embed locally, upsert into Qdrant:
python scripts/ingest.py DATA/true_data --collection rag_documents

# Long/noisy corpora: Docling leaks memory across files within one process
# and eventually dies. --auto-resume restarts it in a fresh process on crash,
# replaying already-finished files from the checkpoint cache each time:
python scripts/ingest.py DATA --auto-resume
```

Run `python scripts/ingest.py --help` for the full flag list (chunk size/overlap,
checkpoint directory, disabling the cache, max resume attempts).

Ingestion is idempotent per file: each file's parsed+chunked output is cached
in `.ingest_cache/` (see `rag_app/ingestion/checkpoint.py`), keyed by content
hash and chunk parameters, so re-running a folder only redoes work for files
that changed or weren't finished last time.

## Tests

```bash
pytest
```

Test layout mirrors `rag_app/`: `tests/ingestion/`, `tests/vectorstore/`,
`tests/scripts/`.

## Data

`DATA/true_data/` holds real project documents used as retrieval targets.
`DATA/noisy_data/` is a synthetic corpus of unrelated public documents (see
`DATA/noisy_data/README.txt`) used to test parsing/ingestion at scale without
exercising real content.
