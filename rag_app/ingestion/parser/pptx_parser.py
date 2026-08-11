"""PPTX parser using slide XML files."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from rag_app.ingestion.models import ParseResult
from rag_app.ingestion.parser.base import BaseParser
from rag_app.ingestion.parser.docling_utils import convert_with_docling


DRAWING_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
SLIDE_RE = re.compile(r"ppt/slides/slide(\d+)\.xml$")


class PptxParser(BaseParser):
    parser_name = "pptx"
    supported_extensions = {".pptx"}

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
            with zipfile.ZipFile(path) as archive:
                slide_names = sorted(
                    (
                        name
                        for name in archive.namelist()
                        if SLIDE_RE.match(name)
                    ),
                    key=_slide_sort_key,
                )
                slide_texts = [
                    _extract_slide_text(archive.read(name), index)
                    for index, name in enumerate(slide_names, start=1)
                ]
        except Exception as exc:
            return ParseResult(
                document=None,
                parser_name=self.parser_name,
                errors=[f"Unable to open PPTX package: {exc}"],
            )

        content = "\n\n".join(text for text in slide_texts if text.strip())
        return ParseResult(
            document=self.make_document(
                path,
                content,
                metadata={"slide_count": len(slide_names)},
            ),
            parser_name=self.parser_name,
        )


def _slide_sort_key(name: str) -> int:
    match = SLIDE_RE.match(name)
    return int(match.group(1)) if match else 0


def _extract_slide_text(xml_bytes: bytes, slide_number: int) -> str:
    root = ElementTree.fromstring(xml_bytes)
    texts = [
        node.text or ""
        for node in root.iter()
        if node.tag == f"{DRAWING_NS}t" and node.text
    ]
    body = "\n".join(text.strip() for text in texts if text.strip())
    return f"[Slide {slide_number}]\n{body}" if body else ""
