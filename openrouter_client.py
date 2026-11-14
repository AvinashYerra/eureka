from __future__ import annotations

import json
import logging
import os
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openrouter/free"
CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 90.0
APP_TITLE = "Eureka"

Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]
ENV_FILE = Path(__file__).resolve().parent / ".env"
SECRET_ENV_KEYS = ("OPENROUTER_API_KEY", "OPENROUTER_MODEL")


def _load_streamlit_secrets() -> None:
    try:
        import streamlit as st
    except Exception:
        return

    try:
        secrets = st.secrets
    except Exception:
        return

    for key in SECRET_ENV_KEYS:
        try:
            value = secrets.get(key)
        except Exception:
            continue
        if value is not None and not os.environ.get(key):
            os.environ[key] = str(value)


def load_backend_env(path: str | Path | None = None) -> None:
    _load_streamlit_secrets()
    if load_dotenv is None:
        return
    env_path = Path(path) if path is not None else ENV_FILE
    try:
        load_dotenv(dotenv_path=env_path, override=False, encoding="utf-8")
    except Exception:
        logger.warning("Could not load backend .env file; using process environment")


load_backend_env()

ERROR_MESSAGES = {
    "missing_credentials": (
        "Recommendation extraction is not configured. Add OPENROUTER_API_KEY to "
        "the app's Streamlit secrets or the project .env file, then restart the "
        "app. Audio and transcript processing are still available."
    ),
    "auth_failed": (
        "OpenRouter rejected the API key. Check OPENROUTER_API_KEY and try again."
    ),
    "rate_limited": (
        "OpenRouter is rate-limiting requests right now. Wait a moment and try "
        "again. Audio and transcript processing are still available."
    ),
    "model_unavailable": (
        "The configured OpenRouter model is unavailable. Try again or set "
        "OPENROUTER_MODEL to another model."
    ),
    "insufficient_credits": (
        "OpenRouter reported insufficient credits for this request. Check the "
        "OpenRouter account or use the free model router."
    ),
    "blocked": (
        "The recommendation model refused to process this transcript. Audio and "
        "transcript processing are still available."
    ),
    "timeout": (
        "OpenRouter did not respond in time. Audio and transcript processing are "
        "still available."
    ),
    "network_error": (
        "Could not reach OpenRouter. Check the network connection and try again. "
        "Audio and transcript processing are still available."
    ),
    "truncated": (
        "The recommendation model stopped before finishing its response. Try again."
    ),
    "invalid_response": (
        "OpenRouter returned an invalid response. Audio and transcript processing "
        "are still available."
    ),
    "api_unavailable": (
        "OpenRouter is unavailable right now. Audio and transcript processing are "
        "still available."
    ),
    "invalid_request": (
        "OpenRouter rejected the recommendation request. Check the configured model "
        "and try again."
    ),
    "api_error": (
        "Recommendation extraction failed at OpenRouter. Audio and transcript "
        "processing are still available."
    ),
}


class OpenRouterError(Exception):
    def __init__(self, user_message: str, category: str):
        self.user_message = user_message
        self.category = category
        super().__init__(user_message)


def get_model_name() -> str:
    return os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise OpenRouterError(
            ERROR_MESSAGES["missing_credentials"],
            "missing_credentials",
        )
    return key


def _error_message(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
        metadata = error.get("metadata")
        if isinstance(metadata, dict):
            for key in ("raw", "message", "provider_error"):
                value = metadata.get(key)
                if isinstance(value, str):
                    return value
    return ""


def _error_code(payload: dict[str, Any], fallback: int) -> int:
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, int):
            return code
        if isinstance(code, str) and code.isdigit():
            return int(code)
    return fallback


def _category_for_status(status: int, message: str) -> str:
    lowered = message.casefold()
    if status == 401:
        return "auth_failed"
    if status == 402:
        return "insufficient_credits"
    if status == 403:
        if any(token in lowered for token in ("credit", "quota", "billing")):
            return "insufficient_credits"
        if any(token in lowered for token in ("moderation", "safety", "content")):
            return "blocked"
        return "auth_failed"
    if status == 404:
        return "model_unavailable"
    if status == 408:
        return "timeout"
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "api_unavailable"
    if status in (400, 422):
        if any(
            token in lowered
            for token in (
                "no endpoints",
                "model",
                "provider",
                "not support",
                "unsupported",
            )
        ):
            return "model_unavailable"
        return "invalid_request"
    return "api_error"


def _raise_for_status(status: int, message: str) -> None:
    category = _category_for_status(status, message)
    logger.warning("OpenRouter request failed status=%s category=%s", status, category)
    raise OpenRouterError(ERROR_MESSAGES[category], category)


def _decode_payload(raw: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return ""


def _post_json(
    url: str,
    headers: dict[str, str],
    body: bytes,
    timeout: float,
) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
        except OSError:
            raw = b""
        finally:
            exc.close()
        return exc.code, raw
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            logger.warning("OpenRouter request timed out")
            raise OpenRouterError(ERROR_MESSAGES["timeout"], "timeout") from exc
        logger.warning("OpenRouter network failure: %s", type(reason).__name__)
        raise OpenRouterError(ERROR_MESSAGES["network_error"], "network_error") from exc
    except (TimeoutError, socket.timeout) as exc:
        logger.warning("OpenRouter request timed out")
        raise OpenRouterError(ERROR_MESSAGES["timeout"], "timeout") from exc
    except OSError as exc:
        logger.warning("OpenRouter network failure: %s", type(exc).__name__)
        raise OpenRouterError(ERROR_MESSAGES["network_error"], "network_error") from exc


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    response_schema: dict[str, Any] | None = None,
    schema_name: str = "response",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    transport: Transport | None = None,
) -> str:
    if not messages:
        raise OpenRouterError(ERROR_MESSAGES["invalid_request"], "invalid_request")

    key = (api_key or _api_key()).strip()
    selected_model = (model or get_model_name()).strip() or DEFAULT_MODEL
    payload: dict[str, Any] = {
        "model": selected_model,
        "messages": messages,
        "temperature": 0.1,
    }
    if response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": response_schema,
            },
        }

    body = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-OpenRouter-Title": APP_TITLE,
        "User-Agent": "Eureka/0.1",
    }

    logger.info(
        "OpenRouter request model=%s messages=%d chars=%d",
        selected_model,
        len(messages),
        len(body),
    )
    send = transport or _post_json
    try:
        status, raw = send(CHAT_COMPLETIONS_URL, headers, body, timeout)
    except OpenRouterError:
        raise
    except Exception as exc:
        logger.warning("OpenRouter transport failure: %s", type(exc).__name__)
        raise OpenRouterError(ERROR_MESSAGES["api_error"], "api_error") from exc

    data = _decode_payload(raw)
    if not isinstance(data, dict):
        if status >= 400:
            _raise_for_status(status, "")
        logger.warning("OpenRouter response was not valid JSON status=%s", status)
        raise OpenRouterError(ERROR_MESSAGES["invalid_response"], "invalid_response")

    error_message = _error_message(data)
    if status >= 400 or data.get("error"):
        error_status = _error_code(data, status if status >= 400 else 502)
        _raise_for_status(error_status, error_message)

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        logger.warning("OpenRouter response contained no choices")
        raise OpenRouterError(ERROR_MESSAGES["invalid_response"], "invalid_response")

    choice = choices[0]
    if not isinstance(choice, dict):
        logger.warning("OpenRouter response contained an invalid choice")
        raise OpenRouterError(ERROR_MESSAGES["invalid_response"], "invalid_response")

    choice_error = choice.get("error")
    if isinstance(choice_error, dict):
        code = choice_error.get("code")
        message = choice_error.get("message")
        _raise_for_status(
            code if isinstance(code, int) else 502,
            message if isinstance(message, str) else "",
        )

    finish_reason = choice.get("finish_reason")
    if finish_reason == "content_filter":
        raise OpenRouterError(ERROR_MESSAGES["blocked"], "blocked")
    if finish_reason == "length":
        raise OpenRouterError(ERROR_MESSAGES["truncated"], "truncated")
    if finish_reason == "error":
        raise OpenRouterError(ERROR_MESSAGES["api_error"], "api_error")

    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    text = _content_text(content)
    if not text:
        logger.warning("OpenRouter returned an empty message")
        raise OpenRouterError(ERROR_MESSAGES["invalid_response"], "invalid_response")

    response_model = data.get("model")
    logger.info(
        "OpenRouter completed model=%s response_chars=%d",
        response_model if isinstance(response_model, str) else selected_model,
        len(text),
    )
    return text
