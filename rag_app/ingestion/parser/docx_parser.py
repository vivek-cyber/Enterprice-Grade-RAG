"""DOCX parser using the Open Packaging Convention XML files."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree

from rag_app.ingestion.models import ParseResult
from rag_app.ingestion.parser.base import BaseParser
from rag_app.ingestion.parser.docling_utils import convert_with_docling


WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class DocxParser(BaseParser):
    """Parses .docx files via Docling, falling back to raw OOXML on failure."""

    parser_name = "docx"
    supported_extensions = {".docx"}

    def parse(self, path: Path) -> ParseResult:
        """Try Docling first, then the OOXML-XML fallback if Docling fails."""

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
        """Convert via Docling for layout-aware extraction (headings, tables)."""

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
        """Extract text directly from word/document.xml when Docling is unavailable."""

        try:
            with zipfile.ZipFile(path) as archive:
                document_xml = archive.read("word/document.xml")
        except Exception as exc:
            return ParseResult(
                document=None,
                parser_name=self.parser_name,
                errors=[f"Unable to open DOCX package: {exc}"],
            )

        try:
            root = ElementTree.fromstring(document_xml)
            lines: list[str] = []
            heading_count = 0
            table_count = 0

            for child in root.iter():
                if child.tag == f"{WORD_NS}p":
                    text = _paragraph_text(child)
                    if text:
                        style = _paragraph_style(child)
                        if style and "heading" in style.lower():
                            heading_count += 1
                            lines.append(f"# {text}")
                        else:
                            lines.append(text)
                elif child.tag == f"{WORD_NS}tbl":
                    table_count += 1

            content = "\n\n".join(lines)
            return ParseResult(
                document=self.make_document(
                    path,
                    content,
                    metadata={
                        "heading_count": heading_count,
                        "table_count": table_count,
                    },
                ),
                parser_name=self.parser_name,
            )
        except Exception as exc:
            return ParseResult(
                document=None,
                parser_name=self.parser_name,
                errors=[f"Unable to parse DOCX XML: {exc}"],
            )


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    texts = [node.text or "" for node in paragraph.iter() if node.tag == f"{WORD_NS}t"]
    return "".join(texts).strip()


def _paragraph_style(paragraph: ElementTree.Element) -> str | None:
    for node in paragraph.iter():
        if node.tag == f"{WORD_NS}pStyle":
            return node.attrib.get(f"{WORD_NS}val")
    return None
