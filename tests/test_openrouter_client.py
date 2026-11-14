import io
import json
import os
import sys
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import openrouter_client
from openrouter_client import (
    CHAT_COMPLETIONS_URL,
    DEFAULT_MODEL,
    OpenRouterError,
    _post_json,
    chat_completion,
    load_backend_env,
)


class FakeTransport:
    def __init__(self, status=200, payload=None, raw=None, error=None):
        self.status = status
        self.payload = payload
        self.raw = raw
        self.error = error
        self.url = None
        self.headers = None
        self.body = None
        self.timeout = None

    def __call__(self, url, headers, body, timeout):
        self.url = url
        self.headers = headers
        self.body = body
        self.timeout = timeout
        if self.error is not None:
            raise self.error
        if self.raw is not None:
            return self.status, self.raw
        return self.status, json.dumps(self.payload).encode("utf-8")


def _completion(content='{"recommendations": []}', model="test/free"):
    return {
        "id": "gen-1",
        "model": model,
        "choices": [
            {
                "finish_reason": "stop",
                "native_finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


def _messages():
    return [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": "Transcript text."},
    ]


SCHEMA = {
    "type": "object",
    "properties": {"recommendations": {"type": "array"}},
    "required": ["recommendations"],
}


class OpenRouterClientTests(unittest.TestCase):
    def setUp(self):
        secrets_patcher = mock.patch.object(
            openrouter_client,
            "_load_streamlit_secrets",
        )
        secrets_patcher.start()
        self.addCleanup(secrets_patcher.stop)

    def test_load_backend_env_reads_key_and_model_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                'OPENROUTER_API_KEY="file-key"\n'
                "OPENROUTER_MODEL=file/model:free\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                load_backend_env(env_path)
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "file-key")
                self.assertEqual(os.environ["OPENROUTER_MODEL"], "file/model:free")

    def test_load_backend_env_does_not_override_process_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("OPENROUTER_API_KEY=file-key\n", encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {"OPENROUTER_API_KEY": "process-key"},
                clear=True,
            ):
                load_backend_env(env_path)
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "process-key")

    def test_load_backend_env_uses_project_env_file_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("OPENROUTER_API_KEY=default-file-key\n", encoding="utf-8")
            with mock.patch.object(openrouter_client, "ENV_FILE", env_path):
                with mock.patch.dict(os.environ, {}, clear=True):
                    load_backend_env()
                    self.assertEqual(
                        os.environ["OPENROUTER_API_KEY"], "default-file-key"
                    )

    def test_load_backend_env_ignores_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {}, clear=True):
                load_backend_env(Path(tmp) / "missing.env")
                self.assertNotIn("OPENROUTER_API_KEY", os.environ)

    def test_loaded_dotenv_key_is_used_by_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "OPENROUTER_API_KEY=file-key\n"
                "OPENROUTER_MODEL=file/model:free\n",
                encoding="utf-8",
            )
            transport = FakeTransport(payload=_completion())
            with mock.patch.dict(os.environ, {}, clear=True):
                load_backend_env(env_path)
                chat_completion(_messages(), transport=transport)

        self.assertEqual(transport.headers["Authorization"], "Bearer file-key")
        self.assertEqual(json.loads(transport.body)["model"], "file/model:free")

    def test_sends_chat_completion_request(self):
        transport = FakeTransport(payload=_completion())
        text = chat_completion(
            _messages(),
            api_key="test-key",
            model=DEFAULT_MODEL,
            response_schema=SCHEMA,
            schema_name="reel_recommendations",
            transport=transport,
        )

        self.assertEqual(text, '{"recommendations": []}')
        self.assertEqual(transport.url, CHAT_COMPLETIONS_URL)
        self.assertEqual(transport.timeout, 90.0)
        self.assertEqual(transport.headers["Authorization"], "Bearer test-key")
        self.assertEqual(transport.headers["Content-Type"], "application/json")
        self.assertEqual(transport.headers["X-OpenRouter-Title"], "Eureka")
        self.assertNotIn("test-key", transport.body.decode("utf-8"))

        payload = json.loads(transport.body)
        self.assertEqual(payload["model"], DEFAULT_MODEL)
        self.assertEqual(payload["model"], "openrouter/free")
        self.assertEqual(payload["messages"], _messages())
        self.assertEqual(payload["temperature"], 0.1)
        self.assertEqual(
            payload["response_format"],
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "reel_recommendations",
                    "strict": True,
                    "schema": SCHEMA,
                },
            },
        )
        self.assertNotIn("models", payload)
        self.assertNotIn("route", payload)
        self.assertNotIn("plugins", payload)

    def test_uses_model_and_key_from_environment(self):
        transport = FakeTransport(payload=_completion())
        with mock.patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "env-key",
                "OPENROUTER_MODEL": "meta-llama/llama-test:free",
            },
            clear=True,
        ):
            chat_completion(_messages(), transport=transport)

        payload = json.loads(transport.body)
        self.assertEqual(payload["model"], "meta-llama/llama-test:free")
        self.assertEqual(transport.headers["Authorization"], "Bearer env-key")

    def test_explicit_model_overrides_environment(self):
        transport = FakeTransport(payload=_completion())
        with mock.patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "env-key",
                "OPENROUTER_MODEL": "meta-llama/llama-test:free",
            },
            clear=True,
        ):
            chat_completion(
                _messages(),
                model="openrouter/free",
                transport=transport,
            )

        self.assertEqual(json.loads(transport.body)["model"], "openrouter/free")

    def test_missing_key_is_reported_without_network(self):
        transport = FakeTransport(payload=_completion())
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(OpenRouterError) as context:
                chat_completion(_messages(), transport=transport)

        self.assertEqual(context.exception.category, "missing_credentials")
        self.assertIn("OPENROUTER_API_KEY", context.exception.user_message)
        self.assertIsNone(transport.url)

    def test_google_key_does_not_enable_openrouter(self):
        transport = FakeTransport(payload=_completion())
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "old-key"}, clear=True):
            with self.assertRaises(OpenRouterError) as context:
                chat_completion(_messages(), transport=transport)

        self.assertEqual(context.exception.category, "missing_credentials")
        self.assertIsNone(transport.url)

    def test_invalid_json_response_is_reported(self):
        transport = FakeTransport(raw=b"not json")
        with self.assertRaises(OpenRouterError) as context:
            chat_completion(_messages(), api_key="key", transport=transport)
        self.assertEqual(context.exception.category, "invalid_response")

    def test_http_429_is_rate_limited(self):
        transport = FakeTransport(
            status=429,
            payload={"error": {"code": 429, "message": "Too many requests"}},
        )
        with self.assertRaises(OpenRouterError) as context:
            chat_completion(_messages(), api_key="key", transport=transport)
        self.assertEqual(context.exception.category, "rate_limited")

    def test_http_401_is_auth_failed(self):
        transport = FakeTransport(
            status=401,
            payload={"error": {"code": 401, "message": "No auth credentials found"}},
        )
        with self.assertRaises(OpenRouterError) as context:
            chat_completion(_messages(), api_key="key", transport=transport)
        self.assertEqual(context.exception.category, "auth_failed")

    def test_http_402_is_insufficient_credits(self):
        transport = FakeTransport(
            status=402,
            payload={"error": {"code": 402, "message": "Insufficient credits"}},
        )
        with self.assertRaises(OpenRouterError) as context:
            chat_completion(_messages(), api_key="key", transport=transport)
        self.assertEqual(context.exception.category, "insufficient_credits")

    def test_http_404_is_model_unavailable(self):
        transport = FakeTransport(
            status=404,
            payload={"error": {"code": 404, "message": "No endpoints found"}},
        )
        with self.assertRaises(OpenRouterError) as context:
            chat_completion(_messages(), api_key="key", transport=transport)
        self.assertEqual(context.exception.category, "model_unavailable")

    def test_http_503_is_api_unavailable(self):
        transport = FakeTransport(
            status=503,
            payload={"error": {"code": 503, "message": "Service unavailable"}},
        )
        with self.assertRaises(OpenRouterError) as context:
            chat_completion(_messages(), api_key="key", transport=transport)
        self.assertEqual(context.exception.category, "api_unavailable")

    def test_http_400_no_endpoints_is_model_unavailable(self):
        transport = FakeTransport(
            status=400,
            payload={
                "error": {
                    "code": 400,
                    "message": "No endpoints found that support structured outputs",
                }
            },
        )
        with self.assertRaises(OpenRouterError) as context:
            chat_completion(_messages(), api_key="key", transport=transport)
        self.assertEqual(context.exception.category, "model_unavailable")

    def test_http_400_invalid_request_is_reported(self):
        transport = FakeTransport(
            status=400,
            payload={"error": {"code": 400, "message": "Invalid request body"}},
        )
        with self.assertRaises(OpenRouterError) as context:
            chat_completion(_messages(), api_key="key", transport=transport)
        self.assertEqual(context.exception.category, "invalid_request")

    def test_top_level_error_with_http_200_is_mapped(self):
        transport = FakeTransport(
            payload={"error": {"code": 429, "message": "Rate limited"}},
        )
        with self.assertRaises(OpenRouterError) as context:
            chat_completion(_messages(), api_key="key", transport=transport)
        self.assertEqual(context.exception.category, "rate_limited")

    def test_choice_error_is_mapped(self):
        payload = _completion()
        payload["choices"][0]["error"] = {"code": 429, "message": "Rate limited"}
        transport = FakeTransport(payload=payload)
        with self.assertRaises(OpenRouterError) as context:
            chat_completion(_messages(), api_key="key", transport=transport)
        self.assertEqual(context.exception.category, "rate_limited")

    def test_missing_choices_is_invalid_response(self):
        transport = FakeTransport(payload={"model": "test/free"})
        with self.assertRaises(OpenRouterError) as context:
            chat_completion(_messages(), api_key="key", transport=transport)
        self.assertEqual(context.exception.category, "invalid_response")

    def test_empty_content_is_invalid_response(self):
        transport = FakeTransport(payload=_completion(content=""))
        with self.assertRaises(OpenRouterError) as context:
            chat_completion(_messages(), api_key="key", transport=transport)
        self.assertEqual(context.exception.category, "invalid_response")

    def test_content_parts_are_joined(self):
        payload = _completion()
        payload["choices"][0]["message"]["content"] = [
            {"type": "text", "text": "{"},
            {"type": "text", "text": '"recommendations":[]}'},
        ]
        transport = FakeTransport(payload=payload)
        text = chat_completion(_messages(), api_key="key", transport=transport)
        self.assertEqual(text, '{"recommendations":[]}')

    def test_length_finish_reason_is_truncated(self):
        payload = _completion()
        payload["choices"][0]["finish_reason"] = "length"
        transport = FakeTransport(payload=payload)
        with self.assertRaises(OpenRouterError) as context:
            chat_completion(_messages(), api_key="key", transport=transport)
        self.assertEqual(context.exception.category, "truncated")

    def test_content_filter_finish_reason_is_blocked(self):
        payload = _completion()
        payload["choices"][0]["finish_reason"] = "content_filter"
        transport = FakeTransport(payload=payload)
        with self.assertRaises(OpenRouterError) as context:
            chat_completion(_messages(), api_key="key", transport=transport)
        self.assertEqual(context.exception.category, "blocked")

    def test_transport_failure_is_mapped_to_api_error(self):
        transport = FakeTransport(error=RuntimeError("boom"))
        with self.assertRaises(OpenRouterError) as context:
            chat_completion(_messages(), api_key="key", transport=transport)
        self.assertEqual(context.exception.category, "api_error")

    def test_network_timeout_is_mapped(self):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(TimeoutError()),
        ):
            with self.assertRaises(OpenRouterError) as context:
                _post_json("https://example.test", {}, b"{}", 1.0)
        self.assertEqual(context.exception.category, "timeout")

    def test_network_failure_is_mapped(self):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(OSError("down")),
        ):
            with self.assertRaises(OpenRouterError) as context:
                _post_json("https://example.test", {}, b"{}", 1.0)
        self.assertEqual(context.exception.category, "network_error")

    def test_http_error_body_is_returned(self):
        body = io.BytesIO(b'{"error":{"code":429,"message":"Too many requests"}}')
        http_error = urllib.error.HTTPError(
            "https://example.test",
            429,
            "Too Many Requests",
            {},
            body,
        )
        with mock.patch("urllib.request.urlopen", side_effect=http_error):
            status, raw = _post_json("https://example.test", {}, b"{}", 1.0)
        http_error.close()
        self.assertEqual(status, 429)
        self.assertIn(b"Too many requests", raw)


class StreamlitSecretsTests(unittest.TestCase):
    def _fake_streamlit(self, secrets):
        return types.SimpleNamespace(secrets=secrets)

    def test_loads_key_and_model_from_streamlit_secrets(self):
        fake_streamlit = self._fake_streamlit(
            {
                "OPENROUTER_API_KEY": "secret-key",
                "OPENROUTER_MODEL": "secret/model:free",
            }
        )
        with mock.patch.dict(sys.modules, {"streamlit": fake_streamlit}):
            with mock.patch.dict(os.environ, {}, clear=True):
                load_backend_env(Path(tempfile.gettempdir()) / "missing.env")
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "secret-key")
                self.assertEqual(
                    os.environ["OPENROUTER_MODEL"],
                    "secret/model:free",
                )

    def test_process_environment_wins_over_streamlit_secrets(self):
        fake_streamlit = self._fake_streamlit(
            {"OPENROUTER_API_KEY": "secret-key"}
        )
        with mock.patch.dict(sys.modules, {"streamlit": fake_streamlit}):
            with mock.patch.dict(
                os.environ,
                {"OPENROUTER_API_KEY": "process-key"},
                clear=True,
            ):
                load_backend_env(Path(tempfile.gettempdir()) / "missing.env")
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "process-key")

    def test_missing_streamlit_secrets_do_not_break_local_env_loading(self):
        empty_streamlit = self._fake_streamlit({})
        with mock.patch.dict(sys.modules, {"streamlit": empty_streamlit}):
            with tempfile.TemporaryDirectory() as tmp:
                env_path = Path(tmp) / ".env"
                env_path.write_text(
                    "OPENROUTER_API_KEY=file-key\n",
                    encoding="utf-8",
                )
                with mock.patch.dict(os.environ, {}, clear=True):
                    load_backend_env(env_path)
                    self.assertEqual(os.environ["OPENROUTER_API_KEY"], "file-key")


if __name__ == "__main__":
    unittest.main()
