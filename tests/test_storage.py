import json
import tempfile
import unittest
import wave
from pathlib import Path

import storage
from storage import (
    StorageError,
    audio_file_path,
    delete_files,
    ensure_data_dirs,
    recommendation_file_path,
    reel_lock,
    relative_audio_path,
    relative_recommendation_path,
    relative_transcript_path,
    save_audio_atomic,
    save_recommendations_atomic,
    save_transcript_atomic,
    transcript_file_path,
)


def _write_wav(path: Path, frames: int = 1600) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * frames)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "data"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_creates_directories_and_paths(self):
        audio_dir, transcript_dir, recommendation_dir, lock_dir = ensure_data_dirs(self.root)
        self.assertTrue(audio_dir.is_dir())
        self.assertTrue(transcript_dir.is_dir())
        self.assertTrue(recommendation_dir.is_dir())
        self.assertTrue(lock_dir.is_dir())
        self.assertEqual(
            audio_file_path("ABC123", self.root),
            audio_dir / "ABC123.wav",
        )
        self.assertEqual(
            transcript_file_path("ABC123", self.root),
            transcript_dir / "ABC123.txt",
        )
        self.assertEqual(relative_audio_path("ABC123"), "data/audio/ABC123.wav")
        self.assertEqual(
            recommendation_file_path("ABC123", self.root),
            recommendation_dir / "ABC123.json",
        )
        self.assertEqual(
            relative_transcript_path("ABC123"),
            "data/transcripts/ABC123.txt",
        )
        self.assertEqual(
            relative_recommendation_path("ABC123"),
            "data/recommendations/ABC123.json",
        )

    def test_saves_audio_atomically(self):
        source = Path(self.temp_dir.name) / "source.wav"
        _write_wav(source)

        final_path = save_audio_atomic(source, "ABC123", self.root)

        self.assertEqual(final_path, self.root.resolve() / "audio" / "ABC123.wav")
        self.assertEqual(final_path.read_bytes(), source.read_bytes())
        self.assertEqual(
            list((self.root / "audio").glob(".ABC123-*")),
            [],
        )

    def test_replaces_transcript_atomically(self):
        first_path = save_transcript_atomic("First transcript.", "ABC123", self.root)
        self.assertEqual(first_path.read_text(encoding="utf-8"), "First transcript.\n")

        second_path = save_transcript_atomic("Updated transcript.", "ABC123", self.root)
        self.assertEqual(second_path.read_text(encoding="utf-8"), "Updated transcript.\n")
        self.assertEqual(
            list((self.root / "transcripts").glob(".ABC123-*")),
            [],
        )

    def test_saves_recommendations_atomically(self):
        payload = {
            "shortcode": "ABC123",
            "recommendations": [{"title": "A video", "search_query": "A video"}],
        }

        path = save_recommendations_atomic(payload, "ABC123", self.root)

        self.assertEqual(path, self.root.resolve() / "recommendations" / "ABC123.json")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)
        self.assertEqual(
            list((self.root / "recommendations").glob(".ABC123-*")),
            [],
        )

    def test_empty_transcript_is_saved_as_empty_file(self):
        path = save_transcript_atomic("", "ABC123", self.root)
        self.assertTrue(path.is_file())
        self.assertEqual(path.read_text(encoding="utf-8"), "")

    def test_delete_files_removes_saved_files(self):
        source = Path(self.temp_dir.name) / "source.wav"
        _write_wav(source)
        audio_path = save_audio_atomic(source, "ABC123", self.root)
        transcript_path = save_transcript_atomic(
            "Watch this video.", "ABC123", self.root
        )

        failed = delete_files(audio_path, transcript_path)

        self.assertEqual(failed, ())
        self.assertFalse(audio_path.exists())
        self.assertFalse(transcript_path.exists())

    def test_delete_files_ignores_missing_and_none_paths(self):
        missing = self.root / "audio" / "missing.wav"

        failed = delete_files(None, missing)

        self.assertEqual(failed, ())

    @unittest.skipIf(storage.fcntl is None, "file locking is unavailable")
    def test_reel_lock_rejects_concurrent_processing(self):
        with reel_lock("ABC123", self.root, blocking=False):
            with self.assertRaises(StorageError) as context:
                with reel_lock("ABC123", self.root, blocking=False):
                    pass
        self.assertEqual(context.exception.category, "busy")
        with reel_lock("ABC123", self.root, blocking=False):
            pass


if __name__ == "__main__":
    unittest.main()
