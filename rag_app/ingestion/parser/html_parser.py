"""HTML parser with basic structural preservation."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from rag_app.ingestion.models import ParseResult
from rag_app.ingestion.parser.base import BaseParser
from rag_app.ingestion.parser.docling_utils import convert_with_docling


BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "caption",
    "div",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}


class _StructuralHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in BLOCK_TAGS:
            self._append_boundary()
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.parts.append(f" [link: {href}] ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in BLOCK_TAGS:
            self._append_boundary()

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        self.parts.append(text)

    def _append_boundary(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    @property
    def text(self) -> str:
        return " ".join(self.parts)

    @property
    def title(self) -> str | None:
        title = " ".join(self.title_parts).strip()
        return title or None


class HtmlParser(BaseParser):
    parser_name = "html"
    supported_extensions = {".html", ".htm"}

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
        raw = path.read_text(encoding="utf-8", errors="replace")
        parser = _StructuralHtmlParser()
        parser.feed(raw)
        parser.close()

        return ParseResult(
            document=self.make_document(
                path,
                parser.text,
                title=parser.title,
                metadata={"html_title": parser.title},
            ),
            parser_name=self.parser_name,
        )
