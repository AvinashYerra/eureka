from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

ALLOWED_HOSTS = frozenset({"www.instagram.com", "instagram.com", "m.instagram.com"})
REEL_PATH_RE = re.compile(r"^/(?:reel|reels)/([A-Za-z0-9_-]{5,30})/?$")

MAX_URL_LENGTH = 2048
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_DURATION_SECONDS = 300
MAX_PROCESSING_SECONDS = 120
SOCKET_TIMEOUT_SECONDS = 20
FFPROBE_TIMEOUT_SECONDS = 15
FFMPEG_TIMEOUT_SECONDS = 60
MAX_TEMP_BYTES = MAX_FILE_SIZE_BYTES * 2 + 10 * 1024 * 1024
AUDIO_SOURCE_SUFFIXES = {".m4a", ".mp4", ".webm", ".mkv", ".mov", ".aac", ".opus", ".ogg", ".wav"}


class ReelValidationError(ValueError):
    pass


class ReelDownloadError(Exception):
    def __init__(self, user_message: str, category: str):
        self.user_message = user_message
        self.category = category
        super().__init__(user_message)


class NoAudioError(ReelDownloadError):
    def __init__(
        self,
        user_message: str = "This Reel has no audio track. Transcription was skipped.",
    ):
        super().__init__(user_message, "no_audio")


@dataclass(frozen=True)
class PreparedAudio:
    shortcode: str
    wav_path: Path
    temp_dir: Path
    duration_seconds: float
    size_bytes: int
    source_was_video: bool

    def cleanup(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def validate_reel_url(raw_url: str) -> tuple[str, str]:
    if not isinstance(raw_url, str):
        raise ReelValidationError("Enter a public Instagram Reel URL.")

    url = raw_url.strip()
    if not url or len(url) > MAX_URL_LENGTH:
        raise ReelValidationError("Enter a valid public Instagram Reel URL.")
    if any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise ReelValidationError("The URL contains invalid characters.")

    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise ReelValidationError("Enter a valid public Instagram Reel URL.") from exc

    if parsed.scheme.lower() not in {"http", "https"}:
        raise ReelValidationError("Only http and https Instagram Reel URLs are supported.")
    if parsed.username or parsed.password:
        raise ReelValidationError("The URL must not contain credentials.")
    if parsed.hostname is None or parsed.hostname.lower() not in ALLOWED_HOSTS:
        raise ReelValidationError("Only Instagram Reel URLs are supported.")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ReelValidationError("The URL contains an invalid port.") from exc
    if port is not None:
        raise ReelValidationError("The URL must use the standard HTTPS port.")

    match = REEL_PATH_RE.fullmatch(parsed.path)
    if not match:
        raise ReelValidationError(
            "Use a direct Reel URL such as https://www.instagram.com/reel/SHORTCODE/."
        )

    shortcode = match.group(1)
    return f"https://www.instagram.com/reel/{shortcode}/", shortcode


def error_from_stderr(stderr: str) -> ReelDownloadError:
    text = (stderr or "").lower()
    mappings = [
        (
            ("empty media response",),
            "This Reel is unavailable, private, or requires login. Instagram may also "
            "be blocking downloads from this server. Only public Reels can be processed.",
            "unavailable_or_login",
        ),
        (
            ("unable to download video data", "unable to extract", "failed to extract", "unable to download webpage"),
            "Instagram did not allow this Reel to be downloaded from the server. It may "
            "be private, unavailable, or blocked for this environment.",
            "unavailable_or_login",
        ),
        (
            ("content isn't available to everyone", "age-restricted", "age restricted"),
            "This Reel is age-restricted or limited to certain audiences and cannot be downloaded anonymously.",
            "age_restricted",
        ),
        (
            ("rate-limit", "rate limit", "too many requests", "http error 429"),
            "Instagram is rate-limiting requests from this server right now. Try again later.",
            "rate_limited",
        ),
        (
            ("login required", "login_required", "challenge_required", "checkpoint", "sign in"),
            "Instagram requires login for this Reel, so it cannot be downloaded from this server.",
            "login_required",
        ),
        (
            ("private",),
            "This Reel is from a private account.",
            "private",
        ),
        (
            ("unsupported url",),
            "Unsupported URL. Provide a public Instagram Reel URL.",
            "unsupported_url",
        ),
        (
            ("requested format is not available", "no video formats found"),
            "No downloadable media format was found for this Reel.",
            "no_format",
        ),
        (
            ("max-filesize", "larger than"),
            f"The downloadable media exceeds the {MAX_FILE_SIZE_MB} MB size limit.",
            "too_large",
        ),
        (
            ("ffmpeg", "ffprobe"),
            "Server media processor is unavailable.",
            "ffmpeg_missing",
        ),
        (
            ("http error 403", "403 forbidden", "forbidden"),
            "Instagram denied the download request. This can happen when Instagram "
            "blocks cloud server IPs. Try again later.",
            "access_denied",
        ),
        (
            ("http error 404", "http error 410", "not available", "unavailable", "removed", "deleted"),
            "This Reel is unavailable or has been removed.",
            "unavailable",
        ),
        (
            ("timed out", "timeout"),
            "The request timed out. Try again later.",
            "timeout",
        ),
    ]

    for needles, message, category in mappings:
        if any(needle in text for needle in needles):
            return ReelDownloadError(message, category)
    return ReelDownloadError(
        "Download failed. The Reel may be unavailable or private, or Instagram may "
        "be blocking downloads from this server.",
        "unknown",
    )


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ReelDownloadError("Server media processor is unavailable.", "ffmpeg_missing")
    return path


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _temp_usage_exceeds(temp_dir: Path) -> bool:
    total = 0
    largest = 0
    for path in temp_dir.rglob("*"):
        if not path.is_file() or path.name == "yt-dlp.err":
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        total += size
        largest = max(largest, size)
    return largest > MAX_FILE_SIZE_BYTES + 5 * 1024 * 1024 or total > MAX_TEMP_BYTES


def _probe_source(path: Path) -> dict:
    ffprobe = _require_binary("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,format_name:stream=codec_type,codec_name,width,height,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        logger.warning("ffprobe timed out while inspecting downloaded audio")
        raise ReelDownloadError(
            "The downloaded media could not be inspected in time.",
            "invalid_media",
        ) from exc

    if proc.returncode != 0:
        logger.warning(
            "ffprobe failed while inspecting downloaded audio returncode=%s stderr=%s",
            proc.returncode,
            (proc.stderr or "")[-1000:],
        )
        raise ReelDownloadError(
            "The downloaded media could not be inspected.",
            "invalid_media",
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ReelDownloadError(
            "The downloaded media could not be inspected.",
            "invalid_media",
        ) from exc

    streams = payload.get("streams") or []
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not video_streams and not audio_streams:
        raise ReelDownloadError(
            "The downloaded file contains no media streams.",
            "invalid_media",
        )

    duration_candidates = []
    format_info = payload.get("format") or {}
    for value in [format_info.get("duration")] + [stream.get("duration") for stream in streams]:
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            duration_candidates.append(duration)

    return {
        "duration": max(duration_candidates) if duration_candidates else 0.0,
        "has_video": bool(video_streams),
        "has_audio": bool(audio_streams),
        "format_name": str(format_info.get("format_name") or "").lower(),
    }


def _audio_download_command(url: str, temp_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "yt_dlp",
        "--ignore-config",
        "--no-cache-dir",
        "--no-warnings",
        "--no-playlist",
        "--no-progress",
        "--force-ipv4",
        "--socket-timeout",
        str(SOCKET_TIMEOUT_SECONDS),
        "--retries",
        "1",
        "--fragment-retries",
        "1",
        "--extractor-retries",
        "0",
        "--concurrent-fragments",
        "1",
        "--max-filesize",
        f"{MAX_FILE_SIZE_MB}M",
        "-f",
        "bestaudio/bv*+ba/b",
        "-o",
        str(temp_dir / "source.%(ext)s"),
        url,
    ]


def _run_audio_download(url: str, temp_dir: Path) -> None:
    _require_binary("ffmpeg")
    _require_binary("ffprobe")

    command = _audio_download_command(url, temp_dir)

    error_path = temp_dir / "yt-dlp.err"
    try:
        error_file = open(error_path, "w", encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ReelDownloadError("Could not start the audio download.", "internal") from exc

    try:
        proc = subprocess.Popen(
            command,
            cwd=temp_dir,
            stdout=subprocess.DEVNULL,
            stderr=error_file,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        error_file.close()
        raise ReelDownloadError("Server downloader is unavailable.", "internal") from exc

    started = time.monotonic()
    try:
        while proc.poll() is None:
            if time.monotonic() - started > MAX_PROCESSING_SECONDS:
                _terminate_process(proc)
                raise ReelDownloadError(
                    "The audio download took too long. Try a shorter Reel or try again later.",
                    "timeout",
                )
            if _temp_usage_exceeds(temp_dir):
                _terminate_process(proc)
                raise ReelDownloadError(
                    f"The downloadable media exceeds the {MAX_FILE_SIZE_MB} MB size limit.",
                    "too_large",
                )
            time.sleep(0.2)
    finally:
        if proc.poll() is None:
            _terminate_process(proc)
        error_file.close()

    try:
        stderr = error_path.read_text(encoding="utf-8", errors="replace")[-16000:]
    except OSError:
        stderr = ""

    if proc.returncode != 0:
        download_error = error_from_stderr(stderr)
        if download_error.category == "no_format":
            download_error = ReelDownloadError(
                "No downloadable audio was found for this Reel.",
                "no_format",
            )
        logger.warning(
            "yt-dlp audio retrieval failed category=%s returncode=%s",
            download_error.category,
            proc.returncode,
        )
        raise download_error


def _select_audio_source(temp_dir: Path) -> tuple[Path, dict]:
    candidates = [
        path
        for path in temp_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_SOURCE_SUFFIXES
    ]
    if not candidates:
        raise ReelDownloadError(
            "No downloadable audio or video was found for this Reel.",
            "no_format",
        )

    probed = []
    probe_failures = 0
    for path in candidates:
        try:
            info = _probe_source(path)
        except ReelDownloadError:
            probe_failures += 1
            continue
        if info["has_audio"]:
            probed.append((path, info))

    if not probed:
        if probe_failures == len(candidates):
            raise ReelDownloadError(
                "The downloaded media could not be inspected.",
                "invalid_media",
            )
        raise NoAudioError()

    audio_only = [item for item in probed if not item[1]["has_video"]]
    pool = audio_only or probed
    return max(pool, key=lambda item: item[0].stat().st_size)


def _convert_audio_to_wav(source_path: Path, wav_path: Path) -> None:
    ffmpeg = _require_binary("ffmpeg")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        logger.warning("ffmpeg audio conversion timed out")
        raise ReelDownloadError(
            "The audio could not be prepared for transcription in time.",
            "ffmpeg_error",
        ) from exc

    if proc.returncode != 0:
        logger.warning(
            "ffmpeg audio conversion failed returncode=%s stderr=%s",
            proc.returncode,
            (proc.stderr or "")[-1200:],
        )
        raise ReelDownloadError(
            "The audio could not be prepared for transcription.",
            "ffmpeg_error",
        )


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
    except (wave.Error, EOFError, OSError) as exc:
        raise ReelDownloadError(
            "The extracted audio could not be validated.",
            "invalid_media",
        ) from exc

    if frames <= 0 or sample_rate <= 0:
        raise NoAudioError(
            "The Reel's audio track is empty. Transcription was skipped."
        )
    return frames / float(sample_rate)


def prepare_reel_audio(raw_url: str) -> PreparedAudio:
    url, shortcode = validate_reel_url(raw_url)
    temp_dir = Path(tempfile.mkdtemp(prefix="eureka-audio-"))

    try:
        _run_audio_download(url, temp_dir)
        source_path, source_info = _select_audio_source(temp_dir)

        source_size = source_path.stat().st_size
        if source_size > MAX_FILE_SIZE_BYTES:
            raise ReelDownloadError(
                f"The downloadable media exceeds the {MAX_FILE_SIZE_MB} MB size limit.",
                "too_large",
            )

        source_duration = float(source_info.get("duration") or 0.0)
        if source_duration > MAX_DURATION_SECONDS:
            raise ReelDownloadError(
                f"The audio exceeds the {MAX_DURATION_SECONDS // 60}-minute duration limit.",
                "too_long",
            )

        wav_path = temp_dir / "audio.wav"
        _convert_audio_to_wav(source_path, wav_path)
        duration_seconds = _wav_duration(wav_path)
        if duration_seconds > MAX_DURATION_SECONDS:
            raise ReelDownloadError(
                f"The audio exceeds the {MAX_DURATION_SECONDS // 60}-minute duration limit.",
                "too_long",
            )

        size_bytes = wav_path.stat().st_size
        if size_bytes <= 44:
            raise NoAudioError(
                "The Reel's audio track is empty. Transcription was skipped."
            )

        logger.info(
            "Prepared audio for %s duration=%.2fs source_video=%s",
            shortcode,
            duration_seconds,
            source_info["has_video"],
        )
        return PreparedAudio(
            shortcode=shortcode,
            wav_path=wav_path,
            temp_dir=temp_dir,
            duration_seconds=round(duration_seconds, 2),
            size_bytes=size_bytes,
            source_was_video=bool(source_info["has_video"]),
        )
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
