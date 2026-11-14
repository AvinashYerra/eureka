import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pipeline
from downloader import NoAudioError
from recommendations import (
    Recommendation,
    RecommendationError,
    RecommendationExtraction,
)
from transcriber import TranscriptionResult
from youtube_search import YouTubeSearchError, YouTubeVideo


class FakePreparedAudio:
    def __init__(self, path: Path):
        self.wav_path = path
        self.source_was_video = False
        self.cleaned_up = False

    def cleanup(self):
        self.cleaned_up = True


def _transcription(text: str = "Watch this video.") -> TranscriptionResult:
    return TranscriptionResult(
        text=text,
        language="en",
        language_probability=0.99,
        duration_seconds=12.0,
        segment_count=1,
    )


def _recommendation() -> Recommendation:
    return Recommendation(
        title="A video",
        search_query="A video tutorial",
        creator_or_channel="A channel",
        category="Tutorial",
        topic="Testing",
        context="Recommended for testing.",
        confidence="high",
        title_type="exact",
    )


def _video() -> YouTubeVideo:
    return YouTubeVideo(
        video_id="abcdefghijk",
        title="A video tutorial",
        channel="A channel",
        url="https://www.youtube.com/watch?v=abcdefghijk",
        thumbnail_url="https://img.example/thumb.jpg",
        duration_seconds=120,
    )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.audio_path = self.root / "audio.wav"
        self.audio_path.write_bytes(b"audio")
        self.transcript_path = self.root / "transcript.txt"
        self.transcript_path.write_text("Watch this video.\n", encoding="utf-8")
        self.recommendation_path = self.root / "recommendations.json"
        self.prepared = FakePreparedAudio(self.audio_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    @contextlib.contextmanager
    def _patches(self, **overrides):
        defaults = {
            "ensure_data_dirs": mock.Mock(),
            "reel_lock": mock.Mock(return_value=contextlib.nullcontext()),
            "prepare_reel_audio": mock.Mock(return_value=self.prepared),
            "save_audio_atomic": mock.Mock(return_value=self.audio_path),
            "transcribe_audio": mock.Mock(return_value=_transcription()),
            "save_transcript_atomic": mock.Mock(return_value=self.transcript_path),
            "extract_recommendations": mock.Mock(
                return_value=RecommendationExtraction(True, (_recommendation(),))
            ),
            "search_youtube": mock.Mock(return_value=(_video(),)),
            "save_recommendations_atomic": mock.Mock(
                return_value=self.recommendation_path
            ),
            "delete_files": mock.Mock(return_value=()),
        }
        defaults.update(overrides)
        with contextlib.ExitStack() as stack:
            mocks = {
                name: stack.enter_context(mock.patch.object(pipeline, name, replacement))
                for name, replacement in defaults.items()
            }
            yield mocks

    def test_processes_audio_to_recommendations(self):
        stages = []
        with self._patches() as mocks:
            result = pipeline.process_reel(
                "https://www.instagram.com/reel/CteWys7p45N/",
                on_stage=stages.append,
            )

        self.assertTrue(result.has_audio)
        self.assertEqual(result.shortcode, "CteWys7p45N")
        self.assertEqual(len(result.recommendations), 1)
        match = result.recommendations[0]
        self.assertEqual(match.recommendation.title, "A video")
        self.assertEqual(match.videos[0].video_id, "abcdefghijk")
        self.assertIsNone(match.search_error)
        self.assertEqual(result.recommendation_error, None)
        self.assertEqual(result.recommendation_path, self.recommendation_path)
        self.assertTrue(result.is_recommendation_reel)
        self.assertIsNone(result.audio_path)
        self.assertIsNone(result.transcript_path)
        self.assertEqual(result.audio_size_bytes, len(b"audio"))
        self.assertIn("Extracting recommendations", stages)
        self.assertIn("Searching YouTube (1/1)", stages)
        self.assertIn("Saving recommendation results", stages)
        self.assertTrue(self.prepared.cleaned_up)

        mocks["delete_files"].assert_called_once_with(
            self.audio_path, self.transcript_path
        )
        mocks["search_youtube"].assert_called_once_with("A video tutorial")
        payload = mocks["save_recommendations_atomic"].call_args.args[0]
        self.assertEqual(payload["shortcode"], "CteWys7p45N")
        self.assertEqual(payload["error"], None)
        self.assertTrue(payload["is_recommendation_reel"])
        self.assertEqual(
            payload["recommendations"][0]["youtube_results"][0]["video_id"],
            "abcdefghijk",
        )

    def test_non_recommendation_reel_skips_youtube_search(self):
        with self._patches(
            extract_recommendations=mock.Mock(
                return_value=RecommendationExtraction(False, ())
            )
        ) as mocks:
            result = pipeline.process_reel(
                "https://www.instagram.com/reel/CteWys7p45N/"
            )

        self.assertTrue(result.has_audio)
        self.assertIs(result.is_recommendation_reel, False)
        self.assertEqual(result.recommendations, ())
        self.assertIsNone(result.recommendation_error)
        self.assertEqual(result.audio_path, self.audio_path)
        self.assertEqual(result.transcript_path, self.transcript_path)
        mocks["search_youtube"].assert_not_called()
        mocks["delete_files"].assert_not_called()
        payload = mocks["save_recommendations_atomic"].call_args.args[0]
        self.assertIs(payload["is_recommendation_reel"], False)
        self.assertEqual(payload["recommendations"], [])

    def test_recommendation_failure_keeps_audio_and_transcript(self):
        with self._patches(
            extract_recommendations=mock.Mock(
                side_effect=RecommendationError(
                    "Recommendation extraction is not configured.",
                    "missing_credentials",
                )
            )
        ) as mocks:
            result = pipeline.process_reel(
                "https://www.instagram.com/reel/CteWys7p45N/"
            )

        self.assertTrue(result.has_audio)
        self.assertEqual(result.transcript, "Watch this video.")
        self.assertEqual(
            result.recommendation_error,
            "Recommendation extraction is not configured.",
        )
        self.assertEqual(result.recommendations, ())
        self.assertIsNone(result.is_recommendation_reel)
        self.assertEqual(result.audio_path, self.audio_path)
        self.assertEqual(result.transcript_path, self.transcript_path)
        mocks["search_youtube"].assert_not_called()
        mocks["delete_files"].assert_not_called()
        payload = mocks["save_recommendations_atomic"].call_args.args[0]
        self.assertEqual(
            payload["error"],
            "Recommendation extraction is not configured.",
        )

    def test_search_failure_is_attached_to_the_recommendation(self):
        with self._patches(
            search_youtube=mock.Mock(
                side_effect=YouTubeSearchError("YouTube search failed.", "search_failed")
            )
        ) as mocks:
            result = pipeline.process_reel(
                "https://www.instagram.com/reel/CteWys7p45N/"
            )

        self.assertEqual(len(result.recommendations), 1)
        self.assertEqual(result.recommendations[0].videos, ())
        self.assertEqual(
            result.recommendations[0].search_error,
            "YouTube search failed.",
        )
        self.assertIsNone(result.recommendation_error)
        self.assertIsNone(result.audio_path)
        self.assertIsNone(result.transcript_path)
        mocks["delete_files"].assert_called_once_with(
            self.audio_path, self.transcript_path
        )

    def test_empty_transcript_skips_extraction_and_search(self):
        with self._patches(
            transcribe_audio=mock.Mock(return_value=_transcription(text=""))
        ) as mocks:
            result = pipeline.process_reel(
                "https://www.instagram.com/reel/CteWys7p45N/"
            )

        self.assertTrue(result.has_audio)
        self.assertEqual(result.transcript, "")
        self.assertEqual(result.recommendations, ())
        mocks["extract_recommendations"].assert_not_called()
        mocks["search_youtube"].assert_not_called()
        mocks["delete_files"].assert_not_called()
        mocks["save_recommendations_atomic"].assert_not_called()
        self.assertIsNone(result.recommendation_path)

    def test_no_audio_skips_transcription(self):
        with self._patches(
            prepare_reel_audio=mock.Mock(side_effect=NoAudioError("No audio")),
        ) as mocks:
            result = pipeline.process_reel(
                "https://www.instagram.com/reel/CteWys7p45N/"
            )

        self.assertFalse(result.has_audio)
        self.assertEqual(result.recommendations, ())
        mocks["transcribe_audio"].assert_not_called()
        mocks["save_recommendations_atomic"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
