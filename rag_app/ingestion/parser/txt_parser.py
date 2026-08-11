"""Plain text parser."""

from __future__ import annotations

from pathlib import Path

from rag_app.ingestion.models import ParseResult
from rag_app.ingestion.parser.base import BaseParser


class TxtParser(BaseParser):
    parser_name = "txt"
    supported_extensions = {".txt"}
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

    def parse(self, path: Path) -> ParseResult:
        warnings: list[str] = []
        last_error: Exception | None = None

        for encoding in self.encodings:
            try:
                content = path.read_text(encoding=encoding)
                if encoding != self.encodings[0]:
                    warnings.append(f"Used fallback encoding: {encoding}")
                return ParseResult(
                    document=self.make_document(
                        path,
                        content,
                        metadata={"encoding": encoding},
                    ),
                    parser_name=self.parser_name,
                    warnings=warnings,
                )
            except UnicodeDecodeError as exc:
                last_error = exc

        return ParseResult(
            document=None,
            parser_name=self.parser_name,
            errors=[f"Unable to decode text file: {last_error}"],
        )
