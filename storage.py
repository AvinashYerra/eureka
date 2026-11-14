from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


class StorageError(Exception):
    def __init__(self, user_message: str, category: str = "storage_error"):
        self.user_message = user_message
        self.category = category
        super().__init__(user_message)


def _resolve_root(data_root: Path | None) -> Path:
    return Path(data_root).resolve() if data_root is not None else DEFAULT_DATA_ROOT


def ensure_data_dirs(data_root: Path | None = None) -> tuple[Path, Path, Path, Path]:
    root = _resolve_root(data_root)
    audio_dir = root / "audio"
    transcript_dir = root / "transcripts"
    recommendation_dir = root / "recommendations"
    lock_dir = root / ".locks"
    try:
        audio_dir.mkdir(parents=True, exist_ok=True)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        recommendation_dir.mkdir(parents=True, exist_ok=True)
        lock_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageError("The local data directory could not be created.") from exc
    return audio_dir, transcript_dir, recommendation_dir, lock_dir


def audio_file_path(shortcode: str, data_root: Path | None = None) -> Path:
    audio_dir, _, _, _ = ensure_data_dirs(data_root)
    return audio_dir / f"{shortcode}.wav"


def transcript_file_path(shortcode: str, data_root: Path | None = None) -> Path:
    _, transcript_dir, _, _ = ensure_data_dirs(data_root)
    return transcript_dir / f"{shortcode}.txt"


def recommendation_file_path(shortcode: str, data_root: Path | None = None) -> Path:
    _, _, recommendation_dir, _ = ensure_data_dirs(data_root)
    return recommendation_dir / f"{shortcode}.json"


def relative_audio_path(shortcode: str) -> str:
    return f"data/audio/{shortcode}.wav"


def relative_transcript_path(shortcode: str) -> str:
    return f"data/transcripts/{shortcode}.txt"


def relative_recommendation_path(shortcode: str) -> str:
    return f"data/recommendations/{shortcode}.json"


@contextlib.contextmanager
def reel_lock(
    shortcode: str,
    data_root: Path | None = None,
    blocking: bool = False,
):
    _, _, _, lock_dir = ensure_data_dirs(data_root)
    lock_path = lock_dir / f"{shortcode}.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as exc:
        raise StorageError("The Reel could not be locked for processing.") from exc

    locked = False
    try:
        if fcntl is not None:
            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(fd, flags)
            except BlockingIOError as exc:
                raise StorageError(
                    "This Reel is already being processed. Try again in a moment.",
                    "busy",
                ) from exc
            locked = True
        yield
    finally:
        if locked and fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


def delete_files(*paths: Path | None) -> tuple[Path, ...]:
    failed: list[Path] = []
    for path in paths:
        if path is None:
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            failed.append(Path(path))
    return tuple(failed)


def save_audio_atomic(
    source_wav: Path,
    shortcode: str,
    data_root: Path | None = None,
) -> Path:
    audio_dir, _, _, _ = ensure_data_dirs(data_root)
    final_path = audio_dir / f"{shortcode}.wav"
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{shortcode}-",
            suffix=".wav.tmp",
            dir=audio_dir,
        )
        os.close(fd)
    except OSError as exc:
        raise StorageError("The audio could not be saved locally.") from exc

    try:
        shutil.copyfile(source_wav, temp_name)
        os.replace(temp_name, final_path)
    except OSError as exc:
        raise StorageError("The audio could not be saved locally.") from exc
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)

    return final_path


def save_transcript_atomic(
    text: str,
    shortcode: str,
    data_root: Path | None = None,
) -> Path:
    _, transcript_dir, _, _ = ensure_data_dirs(data_root)
    final_path = transcript_dir / f"{shortcode}.txt"
    normalized = (text or "").strip()
    content = f"{normalized}\n" if normalized else ""

    temp_name: str | None = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{shortcode}-",
            suffix=".txt.tmp",
            dir=transcript_dir,
        )
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as temp_file:
            temp_file.write(content)
        os.replace(temp_name, final_path)
    except OSError as exc:
        raise StorageError("The transcript could not be saved locally.") from exc
    finally:
        if temp_name is not None:
            with contextlib.suppress(OSError):
                os.unlink(temp_name)

    return final_path


def save_recommendations_atomic(
    payload: dict[str, Any],
    shortcode: str,
    data_root: Path | None = None,
) -> Path:
    _, _, recommendation_dir, _ = ensure_data_dirs(data_root)
    final_path = recommendation_dir / f"{shortcode}.json"
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    temp_name: str | None = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{shortcode}-",
            suffix=".json.tmp",
            dir=recommendation_dir,
        )
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as temp_file:
            temp_file.write(content)
        os.replace(temp_name, final_path)
    except OSError as exc:
        raise StorageError("The recommendations could not be saved locally.") from exc
    finally:
        if temp_name is not None:
            with contextlib.suppress(OSError):
                os.unlink(temp_name)

    return final_path
