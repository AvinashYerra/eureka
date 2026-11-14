from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from downloader import NoAudioError, prepare_reel_audio, validate_reel_url
from recommendations import (
    Recommendation,
    RecommendationError,
    extract_recommendations,
)
from storage import (
    StorageError,
    delete_files,
    ensure_data_dirs,
    reel_lock,
    relative_audio_path,
    save_audio_atomic,
    save_recommendations_atomic,
    save_transcript_atomic,
)
from transcriber import TranscriptionError, transcribe_audio
from youtube_search import YouTubeSearchError, YouTubeVideo, search_youtube

logger = logging.getLogger(__name__)

StageCallback = Callable[[str], None] | None


@dataclass(frozen=True)
class RecommendationMatch:
    recommendation: Recommendation
    videos: tuple[YouTubeVideo, ...]
    search_error: str | None = None


@dataclass(frozen=True)
class ProcessedReel:
    shortcode: str
    has_audio: bool
    audio_path: Path | None
    transcript_path: Path | None
    transcript: str
    duration_seconds: float | None
    audio_size_bytes: int | None
    language: str | None
    language_probability: float | None
    source_was_video: bool
    elapsed_seconds: float
    recommendations: tuple[RecommendationMatch, ...] = ()
    recommendation_error: str | None = None
    recommendation_path: Path | None = None
    is_recommendation_reel: bool | None = None


def _notify(on_stage: StageCallback, label: str) -> None:
    if on_stage is not None:
        on_stage(label)


def _video_payload(video: YouTubeVideo) -> dict[str, object]:
    return {
        "video_id": video.video_id,
        "title": video.title,
        "channel": video.channel,
        "url": video.url,
        "thumbnail_url": video.thumbnail_url,
        "duration_seconds": video.duration_seconds,
    }


def _recommendation_payload(
    shortcode: str,
    language: str | None,
    matches: list[RecommendationMatch],
    error: str | None,
    is_recommendation_reel: bool | None,
) -> dict[str, object]:
    recommendations = []
    for match in matches:
        recommendation = match.recommendation
        recommendations.append(
            {
                "title": recommendation.title,
                "search_query": recommendation.search_query,
                "creator_or_channel": recommendation.creator_or_channel,
                "category": recommendation.category,
                "topic": recommendation.topic,
                "context": recommendation.context,
                "confidence": recommendation.confidence,
                "title_type": recommendation.title_type,
                "youtube_results": [
                    _video_payload(video) for video in match.videos
                ],
                "search_error": match.search_error,
            }
        )
    return {
        "shortcode": shortcode,
        "language": language,
        "is_recommendation_reel": is_recommendation_reel,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recommendations": recommendations,
        "error": error,
    }


def _extract_and_search(
    transcript: str,
    language: str | None,
    on_stage: StageCallback,
) -> tuple[list[RecommendationMatch], str | None, bool | None]:
    if not transcript:
        return [], None, None

    _notify(on_stage, "Extracting recommendations")
    try:
        extraction = extract_recommendations(transcript, language=language)
    except RecommendationError as exc:
        logger.warning("Recommendation extraction failed category=%s", exc.category)
        return [], exc.user_message, None
    except Exception as exc:
        logger.error(
            "Unexpected recommendation extraction failure: %s",
            type(exc).__name__,
        )
        return [], (
            "Recommendation extraction is unavailable right now. "
            "Audio and transcript processing are still available."
        ), None

    if not extraction.is_recommendation_reel:
        _notify(on_stage, "Not a recommendation Reel")
        return [], None, False

    recommendations = extraction.recommendations
    if not recommendations:
        _notify(on_stage, "No recommendations detected")
        return [], None, True

    matches = []
    for index, recommendation in enumerate(recommendations, start=1):
        _notify(on_stage, f"Searching YouTube ({index}/{len(recommendations)})")
        try:
            videos = search_youtube(recommendation.search_query)
        except YouTubeSearchError as exc:
            logger.warning("YouTube search failed category=%s", exc.category)
            videos = ()
            search_error = exc.user_message
        except Exception as exc:
            logger.error("Unexpected YouTube search failure: %s", type(exc).__name__)
            videos = ()
            search_error = "YouTube search failed. Try again later."
        else:
            search_error = None
        matches.append(
            RecommendationMatch(
                recommendation=recommendation,
                videos=videos,
                search_error=search_error,
            )
        )
    return matches, None, True


def process_reel(raw_url: str, on_stage: StageCallback = None) -> ProcessedReel:
    started = time.perf_counter()
    url, shortcode = validate_reel_url(raw_url)
    ensure_data_dirs()

    with reel_lock(shortcode):
        prepared = None
        try:
            _notify(on_stage, "Retrieving Reel audio")
            prepared = prepare_reel_audio(url)

            _notify(on_stage, "Saving audio locally")
            audio_path = save_audio_atomic(prepared.wav_path, shortcode)

            _notify(on_stage, "Transcribing audio")
            try:
                transcription = transcribe_audio(audio_path)
            except TranscriptionError as exc:
                logger.warning(
                    "Transcription failed for %s category=%s",
                    shortcode,
                    exc.category,
                )
                raise TranscriptionError(
                    f"{exc.user_message} The audio was saved at "
                    f"{relative_audio_path(shortcode)}.",
                    exc.category,
                ) from exc

            transcript_path = save_transcript_atomic(transcription.text, shortcode)
            _notify(on_stage, "Transcription completed")

            matches, recommendation_error, is_recommendation_reel = (
                _extract_and_search(
                    transcription.text,
                    transcription.language,
                    on_stage,
                )
            )

            recommendation_path = None
            if transcription.text:
                _notify(on_stage, "Saving recommendation results")
                payload = _recommendation_payload(
                    shortcode,
                    transcription.language,
                    matches,
                    recommendation_error,
                    is_recommendation_reel,
                )
                try:
                    recommendation_path = save_recommendations_atomic(payload, shortcode)
                except StorageError as exc:
                    logger.warning("Recommendation results could not be saved")
                    recommendation_error = recommendation_error or exc.user_message

            audio_size_bytes = audio_path.stat().st_size
            if recommendation_path is not None and matches:
                failed_deletions = delete_files(audio_path, transcript_path)
                if failed_deletions:
                    logger.warning(
                        "Could not delete temporary files for %s: %s",
                        shortcode,
                        ", ".join(path.name for path in failed_deletions),
                    )
                else:
                    audio_path = None
                    transcript_path = None

            elapsed = round(time.perf_counter() - started, 2)
            logger.info("Processed Reel %s in %.2fs", shortcode, elapsed)
            return ProcessedReel(
                shortcode=shortcode,
                has_audio=True,
                audio_path=audio_path,
                transcript_path=transcript_path,
                transcript=transcription.text,
                duration_seconds=transcription.duration_seconds,
                audio_size_bytes=audio_size_bytes,
                language=transcription.language,
                language_probability=transcription.language_probability,
                source_was_video=prepared.source_was_video,
                elapsed_seconds=elapsed,
                recommendations=tuple(matches),
                recommendation_error=recommendation_error,
                recommendation_path=recommendation_path,
                is_recommendation_reel=is_recommendation_reel,
            )
        except NoAudioError:
            _notify(on_stage, "No audio track found")
            logger.info("Reel %s has no audio track; transcription skipped", shortcode)
            return ProcessedReel(
                shortcode=shortcode,
                has_audio=False,
                audio_path=None,
                transcript_path=None,
                transcript="",
                duration_seconds=None,
                audio_size_bytes=None,
                language=None,
                language_probability=None,
                source_was_video=False,
                elapsed_seconds=round(time.perf_counter() - started, 2),
            )
        finally:
            if prepared is not None:
                prepared.cleanup()
