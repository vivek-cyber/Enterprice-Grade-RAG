"""Parser registry for supported ingestion file types."""

from rag_app.ingestion.parser.base import BaseParser
from rag_app.ingestion.parser.docx_parser import DocxParser
from rag_app.ingestion.parser.html_parser import HtmlParser
from rag_app.ingestion.parser.pdf_parser import PdfParser
from rag_app.ingestion.parser.pptx_parser import PptxParser
from rag_app.ingestion.parser.txt_parser import TxtParser


DEFAULT_PARSERS: tuple[BaseParser, ...] = (
    TxtParser(),
    HtmlParser(),
    PdfParser(),
    DocxParser(),
    PptxParser(),
)


def build_parser_registry(
    parsers: tuple[BaseParser, ...] | None = None,
) -> dict[str, BaseParser]:
    """Return a normalized extension-to-parser mapping."""

    registry: dict[str, BaseParser] = {}
    for parser in parsers or DEFAULT_PARSERS:
        for extension in parser.supported_extensions:
            registry[extension.lower()] = parser
    return registry


__all__ = [
    "BaseParser",
    "DEFAULT_PARSERS",
    "build_parser_registry",
    "TxtParser",
    "HtmlParser",
    "PdfParser",
    "DocxParser",
    "PptxParser",
]
