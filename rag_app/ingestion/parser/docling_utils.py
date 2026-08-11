"""Optional Docling integration helpers."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Any


class DoclingUnavailableError(RuntimeError):
    """Raised when Docling is not installed in the active environment."""


class DoclingConversionError(RuntimeError):
    """Raised when Docling is installed but cannot convert a document."""


def convert_with_docling(path: Path) -> tuple[str, dict[str, Any]]:
    """Convert a document with Docling and return Markdown plus metadata."""

    try:
        document_converter = import_module("docling.document_converter")
    except ModuleNotFoundError as exc:
        raise DoclingUnavailableError(
            "Docling is not installed in the active environment."
        ) from exc

    try:
        converter = _build_converter(path, document_converter)
        result = converter.convert(str(path))
        document = result.document
        markdown = _export_markdown(document)
        metadata = _docling_metadata(result, document)
    except Exception as exc:
        raise DoclingConversionError(f"Docling conversion failed: {exc}") from exc

    if not markdown.strip():
        raise DoclingConversionError("Docling returned empty document content.")

    return markdown, metadata


def _build_converter(path: Path, document_converter: Any) -> Any:
    if path.suffix.lower() != ".pdf":
        return document_converter.DocumentConverter()

    base_models = import_module("docling.datamodel.base_models")
    pipeline_options = import_module("docling.datamodel.pipeline_options")

    pdf_options = pipeline_options.PdfPipelineOptions()
    artifacts_path = _resolve_artifacts_path()
    if artifacts_path is not None:
        pdf_options.artifacts_path = artifacts_path
    pdf_options.do_ocr = False
    pdf_options.do_table_structure = True
    pdf_options.do_picture_classification = True
    pdf_options.do_picture_description = True
    pdf_options.generate_picture_images = True
    pdf_options.generate_page_images = False

    return document_converter.DocumentConverter(
        format_options={
            base_models.InputFormat.PDF: document_converter.PdfFormatOption(
                pipeline_options=pdf_options
            )
        }
    )


def _resolve_artifacts_path() -> Path | None:
    configured_path = os.getenv("DOCLING_ARTIFACTS_PATH")
    if configured_path:
        return Path(configured_path)

    project_local_path = Path.cwd() / ".docling-models"
    if project_local_path.exists():
        return project_local_path

    return None


def _export_markdown(document: Any) -> str:
    if hasattr(document, "export_to_markdown"):
        return document.export_to_markdown()
    if hasattr(document, "export_to_text"):
        return document.export_to_text()
    return str(document)


def _docling_metadata(result: Any, document: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {"parser_engine": "docling"}

    for attr_name in ("pages", "tables", "pictures"):
        value = getattr(document, attr_name, None)
        if value is not None:
            try:
                metadata[f"{attr_name}_count"] = len(value)
            except TypeError:
                pass

    input_info = getattr(result, "input", None)
    if input_info is not None:
        metadata["docling_input"] = str(input_info)

    return metadata
