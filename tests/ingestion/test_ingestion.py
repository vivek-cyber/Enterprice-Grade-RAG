from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from rag_app.ingestion.chunking import chunk_document
from rag_app.ingestion.cleaners import clean_text
from rag_app.ingestion.embeddings import EmbeddingRecord
from rag_app.ingestion.file_discovery import discover_files
from rag_app.ingestion.models import Document
from rag_app.ingestion.parser import build_parser_registry
from rag_app.ingestion.parser.docling_utils import convert_with_docling
from rag_app.ingestion.parser.html_parser import HtmlParser
from rag_app.ingestion.parser.pdf_parser import PdfParser
from rag_app.ingestion.pipeline import ingest_folder


ROOT = Path(__file__).resolve().parents[2]
TRUE_DATA = ROOT / "DATA" / "true_data"
NOISY_DATA = ROOT / "DATA" / "noisy_data"


class IngestionTests(unittest.TestCase):
    def test_parser_registry_maps_supported_extensions(self) -> None:
        registry = build_parser_registry()

        for extension in (".txt", ".html", ".pdf", ".docx", ".pptx"):
            self.assertIn(extension, registry)

    def test_file_discovery_classifies_supported_files(self) -> None:
        records = discover_files(TRUE_DATA, set(build_parser_registry()))

        self.assertGreaterEqual(len(records), 6)
        self.assertTrue(all(record.sha256 for record in records))
        self.assertTrue(all(record.supported for record in records))

    def test_txt_html_docx_and_pptx_parse_true_data(self) -> None:
        registry = build_parser_registry()
        samples = [
            TRUE_DATA / "parallel_work_queue.txt",
            TRUE_DATA / "job_management.html",
            TRUE_DATA / "monitor_job.docx",
            TRUE_DATA / "architecture.pptx",
        ]

        for sample in samples:
            with self.subTest(sample=sample.name):
                result = registry[sample.suffix.lower()].parse(sample)
                self.assertEqual([], result.errors)
                self.assertIsNotNone(result.document)
                self.assertTrue(result.document.content.strip())

    def test_docling_primary_path_is_used_when_available(self) -> None:
        parser = HtmlParser()

        with patch(
            "rag_app.ingestion.parser.html_parser.convert_with_docling",
            return_value=("# Converted by Docling", {"parser_engine": "docling"}),
        ):
            result = parser.parse(TRUE_DATA / "job_management.html")

        self.assertEqual([], result.errors)
        self.assertEqual("# Converted by Docling", result.document.content)
        self.assertEqual("docling", result.document.metadata["parser_engine"])

    def test_parser_falls_back_when_docling_fails(self) -> None:
        parser = HtmlParser()

        with patch(
            "rag_app.ingestion.parser.html_parser.convert_with_docling",
            side_effect=RuntimeError("Docling failed"),
        ):
            result = parser.parse(TRUE_DATA / "job_management.html")

        self.assertEqual([], result.errors)
        self.assertTrue(result.document.content.strip())
        self.assertIn("Docling failed", result.warnings)

    def test_docling_pdf_options_include_picture_descriptions(self) -> None:
        captured_options = {}

        class FakePdfPipelineOptions:
            def __init__(self) -> None:
                self.do_ocr = True
                self.do_table_structure = False
                self.do_picture_classification = False
                self.do_picture_description = False
                self.generate_picture_images = False
                self.generate_page_images = True

        class FakePdfFormatOption:
            def __init__(self, pipeline_options) -> None:
                captured_options["pipeline_options"] = pipeline_options

        class FakeDocument:
            pages = [1]
            tables = []
            pictures = [1]

            def export_to_markdown(self) -> str:
                return "# converted"

        class FakeDocumentConverter:
            PdfFormatOption = FakePdfFormatOption

            def __init__(self, format_options=None) -> None:
                captured_options["format_options"] = format_options

            def convert(self, path: str):
                return SimpleNamespace(document=FakeDocument(), input=path)

        def fake_import_module(name: str):
            if name == "docling.document_converter":
                return SimpleNamespace(
                    DocumentConverter=FakeDocumentConverter,
                    PdfFormatOption=FakePdfFormatOption,
                )
            if name == "docling.datamodel.base_models":
                return SimpleNamespace(InputFormat=SimpleNamespace(PDF="pdf"))
            if name == "docling.datamodel.pipeline_options":
                return SimpleNamespace(PdfPipelineOptions=FakePdfPipelineOptions)
            raise ModuleNotFoundError(name)

        with (
            patch("rag_app.ingestion.parser.docling_utils.import_module", fake_import_module),
            patch.dict(
                "os.environ",
                {"DOCLING_ARTIFACTS_PATH": str(ROOT / ".docling-models")},
            ),
        ):
            content, metadata = convert_with_docling(TRUE_DATA / "sample.pdf")

        options = captured_options["pipeline_options"]
        self.assertEqual("# converted", content)
        self.assertEqual("docling", metadata["parser_engine"])
        self.assertFalse(options.do_ocr)
        self.assertTrue(options.do_table_structure)
        self.assertTrue(options.do_picture_classification)
        self.assertTrue(options.do_picture_description)
        self.assertTrue(options.generate_picture_images)
        self.assertFalse(options.generate_page_images)
        self.assertEqual(ROOT / ".docling-models", options.artifacts_path)

    def test_pdf_parser_falls_back_when_docling_fails(self) -> None:
        parser = PdfParser()

        with (
            patch(
                "rag_app.ingestion.parser.pdf_parser.convert_with_docling",
                side_effect=RuntimeError("Docling failed"),
            ),
            patch(
                "rag_app.ingestion.parser.pdf_parser.import_module",
                side_effect=ModuleNotFoundError("pypdf unavailable"),
            ),
        ):
            result = parser.parse(NOISY_DATA / "A Quick Guide To LaTeX.pdf")

        self.assertIsNone(result.document)
        self.assertIn("Docling failed", result.warnings)
        self.assertTrue(any("pypdf" in error for error in result.errors))

    def test_pipeline_records_parser_failures(self) -> None:
        class FailingParser:
            parser_name = "failing-html"
            supported_extensions = {".html"}

            def parse(self, path: Path):
                raise RuntimeError("parser exploded")

        report = ingest_folder(
            TRUE_DATA,
            include_globs=["*.html"],
            parsers=(FailingParser(),),
        )

        self.assertGreater(report.total_files, 0)
        self.assertEqual(report.total_files, report.failed_count)

    def test_cleaning_removes_noise_but_keeps_content(self) -> None:
        cleaned = clean_text("Hello\x00   world\r\n\r\n\r\n    code   line  ")

        self.assertEqual("Hello world\n\n code line", cleaned)

    def test_chunking_creates_ordered_chunks_with_metadata(self) -> None:
        document = Document(
            id="doc-1",
            source_path=TRUE_DATA / "parallel_work_queue.txt",
            source_type=".txt",
            title="Work queue",
            content="A" * 80 + "\n\n" + "B" * 80,
            metadata={"source_name": "parallel_work_queue.txt"},
        )

        chunks = chunk_document(document, chunk_size=70, chunk_overlap=10)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(list(range(len(chunks))), [chunk.chunk_index for chunk in chunks])
        self.assertTrue(all(chunk.metadata["source_path"] for chunk in chunks))
        self.assertTrue(all(chunk.document_id == document.id for chunk in chunks))

    def test_embeddings_contract_imports_without_provider_calls(self) -> None:
        record = EmbeddingRecord(
            chunk_id="chunk-1",
            vector=[0.1, 0.2],
            model="placeholder",
        )

        self.assertEqual("chunk-1", record.chunk_id)


if __name__ == "__main__":
    unittest.main()
