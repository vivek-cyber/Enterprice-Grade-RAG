"""Optional Docling integration helpers."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Any


DOCLING_DEVICE_ENV = "DOCLING_ACCELERATOR_DEVICE"


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
        _reject_incomplete_conversion(result)
        document = result.document
        markdown = _export_markdown(document)
        metadata = _docling_metadata(result, document)
    except DoclingConversionError:
        raise
    except Exception as exc:
        raise DoclingConversionError(f"Docling conversion failed: {exc}") from exc

    if not markdown.strip():
        raise DoclingConversionError("Docling returned empty document content.")

    return markdown, metadata


def _reject_incomplete_conversion(result: Any) -> None:
    """Fail conversions that silently dropped pages.

    Docling does not raise when a page fails mid-pipeline: it flags the page,
    finishes the run, and returns a document containing only the pages that
    survived. Left unchecked that yields a plausible-looking document missing
    most of its content, so treat anything short of full success as a failure
    and let the caller fall back to a whole-text extractor.
    """

    status_name = getattr(getattr(result, "status", None), "name", "")
    if status_name not in {"FAILURE", "PARTIAL_SUCCESS"}:
        return

    failed_pages = sorted(
        {
            page_no
            for error in getattr(result, "errors", []) or []
            if (page_no := getattr(error, "page_no", None)) is not None
        }
    )
    detail = (
        f", {len(failed_pages)} page(s) failed: {_summarize_pages(failed_pages)}"
        if failed_pages
        else ""
    )
    raise DoclingConversionError(f"Docling conversion status {status_name}{detail}")


def _summarize_pages(pages: list[int], *, limit: int = 5) -> str:
    head = ", ".join(str(page) for page in pages[:limit])
    return head if len(pages) <= limit else f"{head}, ... (+{len(pages) - limit} more)"


def _build_converter(path: Path, document_converter: Any) -> Any:
    if path.suffix.lower() != ".pdf":
        return document_converter.DocumentConverter()

    base_models = import_module("docling.datamodel.base_models")
    pipeline_options = import_module("docling.datamodel.pipeline_options")

    pdf_options = pipeline_options.PdfPipelineOptions()
    artifacts_path = _resolve_artifacts_path()
    if artifacts_path is not None:
        pdf_options.artifacts_path = artifacts_path

    # Docling's layout model on CUDA exhausted all 8GB of VRAM on a large PDF
    # and died with cudaErrorUnknown, where the same model on CPU merely runs
    # slower. The GPU is worth far more to the embedding stage, which is the
    # bulk of the work and stable there, so leave Docling on CPU by default.
    accelerator_device = os.getenv(DOCLING_DEVICE_ENV) or "cpu"
    accelerator_options = getattr(pipeline_options, "AcceleratorOptions", None)
    if accelerator_options is not None:
        pdf_options.accelerator_options = accelerator_options(device=accelerator_device)
    pdf_options.do_ocr = False
    pdf_options.do_table_structure = True
    # Picture classification/description load extra vision models and retain
    # full-resolution bitmaps for every figure, which exhausted memory partway
    # through most PDFs (std::bad_alloc in the native page preprocessor) and
    # cost whole pages of text. Author-written "Figure N:" captions live in the
    # text layer and survive without these; table structure is unaffected.
    pdf_options.do_picture_classification = False
    pdf_options.do_picture_description = False
    pdf_options.generate_picture_images = False
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
