from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import quote, urlparse

logger = logging.getLogger(__name__)

SEARCH_LIMIT = 2
SEARCH_TIMEOUT_SECONDS = 30
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}

CommandRunner = Callable[[list[str], int], subprocess.CompletedProcess[str]]


class YouTubeSearchError(Exception):
    def __init__(self, user_message: str, category: str):
        self.user_message = user_message
        self.category = category
        super().__init__(user_message)


@dataclass(frozen=True)
class YouTubeVideo:
    video_id: str
    title: str
    channel: str | None
    url: str
    thumbnail_url: str | None
    duration_seconds: int | None


def _text(value: Any, max_length: int) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    if not cleaned:
        return None
    return cleaned[:max_length]


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_youtube_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in YOUTUBE_HOSTS


def _thumbnail_from_entry(entry: dict[str, Any]) -> str | None:
    thumbnail = _text(entry.get("thumbnail"), 1000)
    if thumbnail and urlparse(thumbnail).scheme == "https":
        return thumbnail

    thumbnails = entry.get("thumbnails")
    if not isinstance(thumbnails, list):
        return None

    candidates = [
        item
        for item in thumbnails
        if isinstance(item, dict)
        and _text(item.get("url"), 1000)
        and urlparse(str(item.get("url"))).scheme == "https"
    ]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda item: _safe_int(item.get("width")) * _safe_int(item.get("height")),
    )
    return _text(best.get("url"), 1000)


def _parse_entry(entry: Any) -> YouTubeVideo | None:
    if not isinstance(entry, dict):
        return None

    video_id = _text(entry.get("id"), 64)
    title = _text(entry.get("title"), 500)
    if not video_id or not title or not re.fullmatch(r"[A-Za-z0-9_-]{6,20}", video_id):
        return None

    url = _text(entry.get("webpage_url") or entry.get("url"), 1000)
    if not url or not _is_youtube_url(url):
        url = f"https://www.youtube.com/watch?v={quote(video_id, safe='-_')}"

    channel = _text(entry.get("channel") or entry.get("uploader"), 200)
    thumbnail_url = _thumbnail_from_entry(entry)

    duration = entry.get("duration")
    duration_seconds = (
        int(round(duration))
        if isinstance(duration, (int, float)) and duration > 0
        else None
    )

    return YouTubeVideo(
        video_id=video_id,
        title=title,
        channel=channel,
        url=url,
        thumbnail_url=thumbnail_url,
        duration_seconds=duration_seconds,
    )


def _default_runner(
    command: list[str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _build_command(query: str, limit: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "yt_dlp",
        "--dump-single-json",
        "--flat-playlist",
        "--no-warnings",
        "--skip-download",
        "--ignore-errors",
        "--playlist-end",
        str(limit),
        "--socket-timeout",
        "15",
        "--retries",
        "1",
        f"ytsearch{limit}:{query}",
    ]


def search_youtube(
    query: str,
    limit: int = SEARCH_LIMIT,
    *,
    runner: CommandRunner | None = None,
) -> tuple[YouTubeVideo, ...]:
    cleaned_query = re.sub(r"\s+", " ", query or "").strip()
    if not cleaned_query:
        raise YouTubeSearchError("The YouTube search query is empty.", "empty_query")
    cleaned_query = cleaned_query[:200]
    limit = max(1, min(int(limit), SEARCH_LIMIT))
    print(f"YouTube search query: {cleaned_query}", flush=True)
    run = runner or _default_runner
    command = _build_command(cleaned_query, limit)

    try:
        completed = run(command, SEARCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        logger.warning("YouTube search timed out")
        raise YouTubeSearchError(
            "YouTube search timed out. Try again later.",
            "timeout",
        ) from exc
    except OSError as exc:
        logger.warning("YouTube search could not start")
        raise YouTubeSearchError(
            "YouTube search could not be started.",
            "search_failed",
        ) from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        logger.warning(
            "YouTube search failed returncode=%s stderr=%s",
            completed.returncode,
            stderr[:300],
        )
        if "no module named yt_dlp" in stderr.casefold():
            raise YouTubeSearchError(
                "YouTube search is not installed on this server.",
                "missing_dependency",
            )
        raise YouTubeSearchError(
            "YouTube search failed. Try again later.",
            "search_failed",
        )

    if not (completed.stdout or "").strip():
        return ()

    try:
        payload = json.loads(completed.stdout or "")
    except json.JSONDecodeError as exc:
        logger.warning("YouTube search returned invalid JSON")
        raise YouTubeSearchError(
            "YouTube returned an unexpected search response.",
            "invalid_response",
        ) from exc

    if not isinstance(payload, dict):
        raise YouTubeSearchError(
            "YouTube returned an unexpected search response.",
            "invalid_response",
        )

    entries = payload.get("entries")
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        raise YouTubeSearchError(
            "YouTube returned an unexpected search response.",
            "invalid_response",
        )

    videos = []
    seen = set()
    for entry in entries:
        video = _parse_entry(entry)
        if video is None or video.video_id in seen:
            continue
        seen.add(video.video_id)
        videos.append(video)
        if len(videos) >= limit:
            break

    return tuple(videos)
