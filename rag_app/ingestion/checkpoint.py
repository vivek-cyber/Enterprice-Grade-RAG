"""Per-file checkpointing for the parse -> clean -> chunk stage.

Parsing is the expensive, restart-hostile part of ingestion: a 60-file run
costs many minutes of CPU, and a crash or kill during embedding used to throw
all of it away. Each file's chunks are written to disk as soon as that file is
chunked, keyed by content hash, so a re-run replays finished files from cache
and only pays for what is genuinely new.

Entries are keyed by the file's sha256 plus the chunk parameters, so editing a
source file or changing chunk sizing naturally misses the cache instead of
serving stale chunks.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rag_app.ingestion.models import Document, DocumentChunk, FileRecord

# Bump when the on-disk entry layout changes so old entries miss the cache
# instead of deserializing into the wrong shape.
CHECKPOINT_SCHEMA_VERSION = 1

DEFAULT_CHECKPOINT_DIRNAME = ".ingest_cache"


@dataclass(slots=True)
class CheckpointEntry:
    """A cached parse+chunk result for a single source file."""

    document: Document
    chunks: list[DocumentChunk]
    warnings: list[str]


class ChunkCheckpointStore:
    """Content-addressed store of per-file parse+chunk results."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, record: FileRecord, *, chunk_size: int, chunk_overlap: int) -> str:
        # The path belongs in the key even though the content hash looks
        # sufficient: document ids and chunk ids are derived from the path, so
        # two byte-identical files at different paths must not share an entry.
        # Keying on content alone made the second file inherit the first one's
        # identity, and the duplicate chunk ids then collapsed on upsert.
        payload = (
            f"{CHECKPOINT_SCHEMA_VERSION}:{record.path.resolve()}:{record.sha256}"
            f":{chunk_size}:{chunk_overlap}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def load(self, key: str) -> CheckpointEntry | None:
        entry_path = self.path_for(key)
        try:
            raw = json.loads(entry_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            # A truncated or unreadable entry is a cache miss, not a hard error;
            # the file simply gets reparsed and the entry rewritten.
            return None

        try:
            return CheckpointEntry(
                document=_document_from_dict(raw["document"]),
                chunks=[_chunk_from_dict(item) for item in raw["chunks"]],
                warnings=list(raw.get("warnings", [])),
            )
        except (KeyError, TypeError):
            return None

    def save(self, key: str, entry: CheckpointEntry) -> None:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "document": _document_to_dict(entry.document),
            "chunks": [_chunk_to_dict(chunk) for chunk in entry.chunks],
            "warnings": entry.warnings,
        }
        _atomic_write_json(self.path_for(key), payload)


def _atomic_write_json(destination: Path, payload: dict) -> None:
    """Write via a temp file + rename so a kill mid-write cannot corrupt an entry."""

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            # default=str keeps parser-supplied metadata (Paths, enums) writable
            # rather than failing the whole entry on one exotic value.
            json.dump(payload, handle, ensure_ascii=False, default=str)
        os.replace(handle.name, destination)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _document_to_dict(document: Document) -> dict:
    return {
        "id": document.id,
        "source_path": str(document.source_path),
        "source_type": document.source_type,
        "title": document.title,
        "content": document.content,
        "metadata": document.metadata,
    }


def _document_from_dict(raw: dict) -> Document:
    return Document(
        id=raw["id"],
        source_path=Path(raw["source_path"]),
        source_type=raw["source_type"],
        title=raw.get("title"),
        content=raw["content"],
        metadata=raw.get("metadata", {}),
    )


def _chunk_to_dict(chunk: DocumentChunk) -> dict:
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "chunk_index": chunk.chunk_index,
        "metadata": chunk.metadata,
    }


def _chunk_from_dict(raw: dict) -> DocumentChunk:
    return DocumentChunk(
        id=raw["id"],
        document_id=raw["document_id"],
        text=raw["text"],
        chunk_index=raw["chunk_index"],
        metadata=raw.get("metadata", {}),
    )
