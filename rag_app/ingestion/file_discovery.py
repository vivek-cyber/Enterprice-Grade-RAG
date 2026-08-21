"""File discovery and hashing for ingestion."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rag_app.ingestion.models import FileRecord


def discover_files(
    source_dir: Path,
    supported_extensions: set[str],
    *,
    include_globs: list[str] | None = None,
) -> list[FileRecord]:
    """Walk a folder and classify files as supported or skipped."""

    source_dir = source_dir.resolve()
    patterns = include_globs or ["**/*"]
    candidates: dict[Path, None] = {}

    for pattern in patterns:
        for path in source_dir.glob(pattern):
            if path.is_file():
                candidates[path] = None

    records: list[FileRecord] = []
    normalized_extensions = {extension.lower() for extension in supported_extensions}

    for path in sorted(candidates):
        extension = path.suffix.lower()
        supported = extension in normalized_extensions
        records.append(
            FileRecord(
                path=path,
                extension=extension,
                sha256=hash_file(path),
                size_bytes=path.stat().st_size,
                supported=supported,
                skip_reason=None
                if supported
                else f"Unsupported extension: {extension}",
            )
        )

    return records


def hash_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the sha256 hex digest of a file's contents, read in fixed-size chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
