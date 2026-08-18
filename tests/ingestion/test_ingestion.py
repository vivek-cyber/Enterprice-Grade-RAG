from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from rag_app.ingestion.checkpoint import CheckpointEntry, ChunkCheckpointStore
from rag_app.ingestion.chunking import chunk_document
from rag_app.ingestion.cleaners import clean_text
from rag_app.ingestion.embeddings import EmbeddingRecord
from rag_app.ingestion.file_discovery import discover_files
from rag_app.ingestion.models import Document, DocumentChunk, FileRecord
from rag_app.ingestion.parser import build_parser_registry
from rag_app.ingestion.parser.docling_utils import (
    DoclingConversionError,
    convert_with_docling,
)
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

    def test_docling_pdf_options_disable_picture_models(self) -> None:
        # Picture classification/description exhausted memory partway through
        # most PDFs and cost whole pages of text, so they stay off; table
        # structure is the structural feature worth paying for.
        captured_options = {}

        class FakePdfPipelineOptions:
            def __init__(self) -> None:
                self.do_ocr = True
                self.do_table_structure = False
                self.do_picture_classification = True
                self.do_picture_description = True
                self.generate_picture_images = True
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
        self.assertFalse(options.do_picture_classification)
        self.assertFalse(options.do_picture_description)
        self.assertFalse(options.generate_picture_images)
        self.assertFalse(options.generate_page_images)
        self.assertEqual(ROOT / ".docling-models", options.artifacts_path)

    def test_docling_partial_success_is_rejected(self) -> None:
        # Docling flags failed pages and still returns a document containing
        # only the survivors. Accepting that silently indexes a truncated file,
        # so a partial conversion must fail loudly enough to trigger fallback.
        class FakeDocument:
            pages = [1]
            tables: list = []
            pictures: list = []

            def export_to_markdown(self) -> str:
                return "# only the first page survived"

        class FakeDocumentConverter:
            def __init__(self, format_options=None) -> None:
                pass

            def convert(self, path: str):
                return SimpleNamespace(
                    document=FakeDocument(),
                    input=path,
                    status=SimpleNamespace(name="PARTIAL_SUCCESS"),
                    errors=[SimpleNamespace(page_no=page) for page in (6, 7, 8)],
                )

        def fake_import_module(name: str):
            if name == "docling.document_converter":
                return SimpleNamespace(
                    DocumentConverter=FakeDocumentConverter,
                    PdfFormatOption=lambda pipeline_options: None,
                )
            if name == "docling.datamodel.base_models":
                return SimpleNamespace(InputFormat=SimpleNamespace(PDF="pdf"))
            if name == "docling.datamodel.pipeline_options":
                return SimpleNamespace(PdfPipelineOptions=SimpleNamespace)
            raise ModuleNotFoundError(name)

        with patch(
            "rag_app.ingestion.parser.docling_utils.import_module", fake_import_module
        ):
            with self.assertRaises(DoclingConversionError) as caught:
                convert_with_docling(TRUE_DATA / "sample.pdf")

        message = str(caught.exception)
        self.assertIn("PARTIAL_SUCCESS", message)
        self.assertIn("3 page(s) failed", message)


class CheckpointTests(unittest.TestCase):
    def _record(self, sha256: str = "abc123", path: str = "doc.pdf") -> FileRecord:
        return FileRecord(
            path=Path(path),
            extension=".pdf",
            sha256=sha256,
            size_bytes=10,
            supported=True,
        )

    def _entry(self) -> CheckpointEntry:
        document = Document(
            id="doc-1",
            source_path=Path("doc.pdf"),
            source_type="pdf",
            title="Doc",
            content="hello world",
            metadata={"pages_count": 2},
        )
        chunk = DocumentChunk(
            id="doc-1-0",
            document_id="doc-1",
            text="hello world",
            chunk_index=0,
            metadata={"source": "doc.pdf"},
        )
        return CheckpointEntry(document=document, chunks=[chunk], warnings=["degraded"])

    def test_checkpoint_round_trips_document_and_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ChunkCheckpointStore(Path(tmp))
            key = store.key(self._record(), chunk_size=1200, chunk_overlap=150)
            store.save(key, self._entry())

            loaded = store.load(key)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual("doc-1", loaded.document.id)
        self.assertEqual(Path("doc.pdf"), loaded.document.source_path)
        self.assertEqual({"pages_count": 2}, loaded.document.metadata)
        self.assertEqual(["degraded"], loaded.warnings)
        self.assertEqual(1, len(loaded.chunks))
        self.assertEqual("hello world", loaded.chunks[0].text)

    def test_checkpoint_key_changes_with_content_and_chunk_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ChunkCheckpointStore(Path(tmp))
            baseline = store.key(self._record(), chunk_size=1200, chunk_overlap=150)

            self.assertNotEqual(
                baseline,
                store.key(self._record("different"), chunk_size=1200, chunk_overlap=150),
            )
            self.assertNotEqual(
                baseline, store.key(self._record(), chunk_size=800, chunk_overlap=150)
            )
            self.assertNotEqual(
                baseline, store.key(self._record(), chunk_size=1200, chunk_overlap=0)
            )
            # Byte-identical files at different paths must not share an entry:
            # document and chunk ids are path-derived, so a shared entry makes
            # the second file inherit the first one's ids and collapse on upsert.
            self.assertNotEqual(
                baseline,
                store.key(
                    self._record(path="other.pdf"), chunk_size=1200, chunk_overlap=150
                ),
            )

    def test_missing_and_corrupt_entries_are_cache_misses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ChunkCheckpointStore(Path(tmp))
            key = store.key(self._record(), chunk_size=1200, chunk_overlap=150)
            self.assertIsNone(store.load(key))

            store.path_for(key).write_text("{ not json", encoding="utf-8")
            self.assertIsNone(store.load(key))

    def test_ingest_folder_replays_files_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ChunkCheckpointStore(Path(tmp))

            first = ingest_folder(TRUE_DATA, checkpoint_store=store)
            self.assertEqual(0, first.cached_files)
            self.assertGreater(first.parsed_files, 0)

            second = ingest_folder(TRUE_DATA, checkpoint_store=store)

        self.assertEqual(first.parsed_files, second.cached_files)
        self.assertEqual(len(first.chunks), len(second.chunks))
        self.assertEqual(
            [chunk.id for chunk in first.chunks], [chunk.id for chunk in second.chunks]
        )

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

    def test_ingest_folder_without_embedding_provider_leaves_embeddings_empty(self) -> None:
        report = ingest_folder(TRUE_DATA)

        self.assertEqual([], report.embeddings)

    def test_ingest_folder_populates_embeddings_when_provider_supplied(self) -> None:
        class FakeEmbeddingProvider:
            provider_name = "fake"
            model_name = "fake-model"

            def embed_texts(
                self, texts: list[str], *, progress_callback=None
            ) -> list[EmbeddingRecord]:
                if progress_callback is not None:
                    progress_callback(len(texts), len(texts))
                return [
                    EmbeddingRecord(chunk_id=str(index), vector=[float(index)], model=self.model_name)
                    for index in range(len(texts))
                ]

        report = ingest_folder(TRUE_DATA, embedding_provider=FakeEmbeddingProvider())

        self.assertEqual(len(report.chunks), len(report.embeddings))
        for chunk, record in zip(report.chunks, report.embeddings, strict=True):
            self.assertEqual(chunk.id, record.chunk_id)

    def test_ingest_folder_raises_on_embedding_count_mismatch(self) -> None:
        class ShortEmbeddingProvider:
            provider_name = "short"
            model_name = "short-model"

            def embed_texts(
                self, texts: list[str], *, progress_callback=None
            ) -> list[EmbeddingRecord]:
                return [EmbeddingRecord(chunk_id="0", vector=[0.0], model=self.model_name)]

        with self.assertRaises(ValueError):
            ingest_folder(TRUE_DATA, embedding_provider=ShortEmbeddingProvider())

    def test_ingest_folder_raises_when_vector_store_given_without_embedding_provider(
        self,
    ) -> None:
        class FakeVectorStore:
            store_name = "fake"

            def ensure_collection(self, vector_size: int) -> None:
                raise AssertionError("should not be called")

            def upsert(self, points) -> None:
                raise AssertionError("should not be called")

            def search(self, vector, *, limit: int = 10):
                raise AssertionError("should not be called")

            def delete(self, chunk_ids) -> None:
                raise AssertionError("should not be called")

        with self.assertRaises(ValueError):
            ingest_folder(TRUE_DATA, vector_store=FakeVectorStore())

    def test_ingest_folder_upserts_into_vector_store_when_supplied(self) -> None:
        class FakeEmbeddingProvider:
            provider_name = "fake"
            model_name = "fake-model"

            def embed_texts(
                self, texts: list[str], *, progress_callback=None
            ) -> list[EmbeddingRecord]:
                if progress_callback is not None:
                    progress_callback(len(texts), len(texts))
                return [
                    EmbeddingRecord(chunk_id=str(index), vector=[float(index)], model=self.model_name)
                    for index in range(len(texts))
                ]

        class FakeVectorStore:
            store_name = "fake"

            def __init__(self) -> None:
                self.ensured_vector_size: int | None = None
                self.upserted_points = []

            def ensure_collection(self, vector_size: int) -> None:
                self.ensured_vector_size = vector_size

            def upsert(self, points) -> None:
                self.upserted_points.extend(points)

            def search(self, vector, *, limit: int = 10):
                raise AssertionError("not exercised in this test")

            def delete(self, chunk_ids) -> None:
                raise AssertionError("not exercised in this test")

        vector_store = FakeVectorStore()
        report = ingest_folder(
            TRUE_DATA,
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=vector_store,
        )

        self.assertEqual(1, vector_store.ensured_vector_size)
        self.assertEqual(len(report.chunks), len(vector_store.upserted_points))
        for chunk, point in zip(report.chunks, vector_store.upserted_points, strict=True):
            self.assertEqual(chunk.id, point.chunk_id)
            self.assertEqual(chunk.text, point.text)


if __name__ == "__main__":
    unittest.main()
