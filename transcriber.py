from __future__ import annotations

import logging
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODEL_NAME = "small"
_model: Any | None = None
_model_lock = threading.Lock()
_cached_load_model: Any | None = None


class TranscriptionError(Exception):
    def __init__(self, user_message: str, category: str):
        self.user_message = user_message
        self.category = category
        super().__init__(user_message)


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    language_probability: float | None
    duration_seconds: float
    segment_count: int


def _validate_wav(audio_path: Path) -> float:
    if not audio_path.is_file():
        raise TranscriptionError(
            "The saved audio file is missing, so transcription was skipped.",
            "audio_missing",
        )

    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
    except (wave.Error, EOFError, OSError) as exc:
        raise TranscriptionError(
            "The saved audio file could not be read.",
            "invalid_audio",
        ) from exc

    if frames <= 0 or sample_rate <= 0:
        raise TranscriptionError(
            "The audio track is empty, so transcription was skipped.",
            "empty_audio",
        )
    return frames / float(sample_rate)


def _load_model() -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "Speech recognition is not installed on this server.",
            "missing_dependency",
        ) from exc

    try:
        return WhisperModel(
            MODEL_NAME,
            device="cpu",
            compute_type="int8",
        )
    except Exception as exc:
        logger.exception("Failed to load faster-whisper model")
        raise TranscriptionError(
            "Speech recognition could not be started.",
            "model_unavailable",
        ) from exc


def _cached_model_loader() -> Any:
    global _cached_load_model
    if _cached_load_model is not None:
        return _cached_load_model

    try:
        import streamlit as st
    except ImportError:
        _cached_load_model = _load_model
    else:
        _cached_load_model = st.cache_resource(show_spinner=False)(_load_model)
    return _cached_load_model


def _get_model() -> Any:
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model
        _model = _cached_model_loader()()

    return _model


def transcribe_audio(audio_path: Path | str) -> TranscriptionResult:
    path = Path(audio_path)
    duration_seconds = _validate_wav(path)
    model = _get_model()

    try:
        segments, info = model.transcribe(
            str(path),
            language=None,
            beam_size=5,
            vad_filter=True,
        )
        texts = []
        for segment in segments:
            segment_text = (segment.text or "").strip()
            if segment_text:
                texts.append(segment_text)
    except Exception as exc:
        logger.exception("Transcription failed for %s", path.name)
        raise TranscriptionError(
            "Transcription failed. Try again later.",
            "transcription_failed",
        ) from exc

    text = " ".join(texts).strip()
    language = getattr(info, "language", None)
    language_probability = getattr(info, "language_probability", None)
    logger.info(
        "Transcribed %s language=%s segments=%d characters=%d",
        path.name,
        language,
        len(texts),
        len(text),
    )
    return TranscriptionResult(
        text=text,
        language=language,
        language_probability=language_probability,
        duration_seconds=round(duration_seconds, 2),
        segment_count=len(texts),
    )
