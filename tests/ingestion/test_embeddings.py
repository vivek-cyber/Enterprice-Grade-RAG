from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from google.genai import errors as genai_errors

from rag_app.ingestion.embeddings.gemini_provider import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIMENSIONALITY,
    DEFAULT_TASK_TYPE,
    GeminiEmbeddingAPIError,
    GeminiEmbeddingConfigError,
    GeminiEmbeddingProvider,
)


def _rate_limit_error() -> genai_errors.APIError:
    return genai_errors.APIError(code=429, response_json={"message": "rate limited"})


def _server_error() -> genai_errors.APIError:
    return genai_errors.APIError(code=500, response_json={"message": "server error"})


class FakeClient:
    def __init__(self, embed_content_fn) -> None:
        self.models = SimpleNamespace(embed_content=embed_content_fn)


def _echo_length_response(*, model, contents, config):
    return SimpleNamespace(
        embeddings=[SimpleNamespace(values=[float(len(text))]) for text in contents]
    )


class GeminiEmbeddingProviderTests(unittest.TestCase):
    def test_missing_api_key_raises_at_construction(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(GeminiEmbeddingConfigError):
                GeminiEmbeddingProvider()

    def test_api_key_from_constructor_overrides_missing_env(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "rag_app.ingestion.embeddings.gemini_provider.genai.Client",
                return_value=FakeClient(_echo_length_response),
            ),
        ):
            provider = GeminiEmbeddingProvider(api_key="explicit-key")

        self.assertEqual("explicit-key", provider.api_key)

    def test_default_model_name_and_task_type(self) -> None:
        with patch(
            "rag_app.ingestion.embeddings.gemini_provider.genai.Client",
            return_value=FakeClient(_echo_length_response),
        ):
            provider = GeminiEmbeddingProvider(api_key="test-key")

        self.assertEqual(DEFAULT_MODEL_NAME, provider.model_name)
        self.assertEqual(DEFAULT_TASK_TYPE, provider.task_type)
        self.assertEqual(DEFAULT_OUTPUT_DIMENSIONALITY, provider.output_dimensionality)

    def test_embed_texts_batches_requests_by_batch_size(self) -> None:
        calls: list[list[str]] = []

        def recording_fn(*, model, contents, config):
            calls.append(list(contents))
            return _echo_length_response(model=model, contents=contents, config=config)

        with patch(
            "rag_app.ingestion.embeddings.gemini_provider.genai.Client",
            return_value=FakeClient(recording_fn),
        ):
            provider = GeminiEmbeddingProvider(api_key="test-key", batch_size=2)
            provider.embed_texts(["a", "bb", "ccc", "dddd", "eeeee"])

        self.assertEqual(
            [["a", "bb"], ["ccc", "dddd"], ["eeeee"]],
            calls,
        )

    def test_embed_texts_preserves_order_across_batches(self) -> None:
        with patch(
            "rag_app.ingestion.embeddings.gemini_provider.genai.Client",
            return_value=FakeClient(_echo_length_response),
        ):
            provider = GeminiEmbeddingProvider(api_key="test-key", batch_size=2)
            records = provider.embed_texts(["a", "bb", "ccc", "dddd", "eeeee"])

        self.assertEqual(
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
            [record.vector for record in records],
        )

    def test_embed_texts_empty_list_returns_empty_without_calling_api(self) -> None:
        calls: list[list[str]] = []

        def recording_fn(*, model, contents, config):
            calls.append(list(contents))
            return _echo_length_response(model=model, contents=contents, config=config)

        with patch(
            "rag_app.ingestion.embeddings.gemini_provider.genai.Client",
            return_value=FakeClient(recording_fn),
        ):
            provider = GeminiEmbeddingProvider(api_key="test-key")
            records = provider.embed_texts([])

        self.assertEqual([], records)
        self.assertEqual(0, len(calls))

    def test_embed_texts_uses_positional_placeholder_chunk_ids(self) -> None:
        with patch(
            "rag_app.ingestion.embeddings.gemini_provider.genai.Client",
            return_value=FakeClient(_echo_length_response),
        ):
            provider = GeminiEmbeddingProvider(api_key="test-key", batch_size=2)
            records = provider.embed_texts(["a", "bb", "ccc"])

        self.assertEqual(["0", "1", "2"], [record.chunk_id for record in records])

    def test_embed_texts_retries_on_429_then_succeeds(self) -> None:
        attempts = {"count": 0}

        def flaky_fn(*, model, contents, config):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise _rate_limit_error()
            return _echo_length_response(model=model, contents=contents, config=config)

        sleep_calls: list[float] = []

        with patch(
            "rag_app.ingestion.embeddings.gemini_provider.genai.Client",
            return_value=FakeClient(flaky_fn),
        ):
            provider = GeminiEmbeddingProvider(
                api_key="test-key",
                _sleep=sleep_calls.append,
            )
            records = provider.embed_texts(["hello"])

        self.assertEqual([[5.0]], [record.vector for record in records])
        self.assertEqual([provider.initial_backoff_seconds], sleep_calls)

    def test_embed_texts_raises_after_max_retries_exhausted(self) -> None:
        attempts = {"count": 0}

        def always_rate_limited(*, model, contents, config):
            attempts["count"] += 1
            raise _rate_limit_error()

        with patch(
            "rag_app.ingestion.embeddings.gemini_provider.genai.Client",
            return_value=FakeClient(always_rate_limited),
        ):
            provider = GeminiEmbeddingProvider(
                api_key="test-key",
                max_retries=2,
                _sleep=lambda seconds: None,
            )
            with self.assertRaises(GeminiEmbeddingAPIError):
                provider.embed_texts(["hello"])

        self.assertEqual(3, attempts["count"])

    def test_embed_texts_raises_immediately_on_non_rate_limit_error(self) -> None:
        attempts = {"count": 0}
        sleep_calls: list[float] = []

        def always_server_error(*, model, contents, config):
            attempts["count"] += 1
            raise _server_error()

        with patch(
            "rag_app.ingestion.embeddings.gemini_provider.genai.Client",
            return_value=FakeClient(always_server_error),
        ):
            provider = GeminiEmbeddingProvider(
                api_key="test-key",
                _sleep=sleep_calls.append,
            )
            with self.assertRaises(GeminiEmbeddingAPIError):
                provider.embed_texts(["hello"])

        self.assertEqual(1, attempts["count"])
        self.assertEqual([], sleep_calls)

    def test_output_dimensionality_and_task_type_passed_through(self) -> None:
        captured_configs = []

        def capturing_fn(*, model, contents, config):
            captured_configs.append(config)
            return _echo_length_response(model=model, contents=contents, config=config)

        with patch(
            "rag_app.ingestion.embeddings.gemini_provider.genai.Client",
            return_value=FakeClient(capturing_fn),
        ):
            provider = GeminiEmbeddingProvider(
                api_key="test-key",
                output_dimensionality=768,
                task_type="RETRIEVAL_QUERY",
            )
            provider.embed_texts(["hello"])

        self.assertEqual(768, captured_configs[0].output_dimensionality)
        self.assertEqual("RETRIEVAL_QUERY", captured_configs[0].task_type)


if __name__ == "__main__":
    unittest.main()
