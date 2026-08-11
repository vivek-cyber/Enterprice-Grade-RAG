"""PDF parser with Docling primary extraction and pypdf fallback."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from rag_app.ingestion.models import ParseResult
from rag_app.ingestion.parser.base import BaseParser
from rag_app.ingestion.parser.docling_utils import convert_with_docling


class PdfParser(BaseParser):
    parser_name = "pdf"
    supported_extensions = {".pdf"}

    def parse(self, path: Path) -> ParseResult:
        docling_result = self.parse_with_docling(path)
        if docling_result.succeeded:
            return docling_result

        fallback_result = self.parse_with_fallback(path)
        fallback_result.warnings = [
            *docling_result.errors,
            *docling_result.warnings,
            *fallback_result.warnings,
        ]
        return fallback_result

    def parse_with_docling(self, path: Path) -> ParseResult:
        try:
            content, metadata = convert_with_docling(path)
            return ParseResult(
                document=self.make_document(path, content, metadata=metadata),
                parser_name=self.parser_name,
            )
        except Exception as exc:
            return ParseResult(
                document=None,
                parser_name=self.parser_name,
                errors=[str(exc)],
            )

    def parse_with_fallback(self, path: Path) -> ParseResult:
        try:
            pypdf = import_module("pypdf")
        except ModuleNotFoundError:
            return ParseResult(
                document=None,
                parser_name=self.parser_name,
                errors=[
                    "PDF parsing requires the optional 'pypdf' package. "
                    "Install it to extract PDF text."
                ],
            )

        try:
            reader = pypdf.PdfReader(str(path))
            page_texts: list[str] = []
            empty_pages: list[int] = []

            for index, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    page_texts.append(f"[Page {index}]\n{text}")
                else:
                    empty_pages.append(index)

            warnings = [
                f"No extractable text on page {page_number}"
                for page_number in empty_pages
            ]
            return ParseResult(
                document=self.make_document(
                    path,
                    "\n\n".join(page_texts),
                    metadata={
                        "page_count": len(reader.pages),
                        "empty_pages": empty_pages,
                    },
                ),
                parser_name=self.parser_name,
                warnings=warnings,
            )
        except Exception as exc:  # pragma: no cover - depends on PDF backend
            return ParseResult(
                document=None,
                parser_name=self.parser_name,
                errors=[f"Unable to parse PDF: {exc}"],
            )
