import tempfile
import unittest
import wave
from pathlib import Path

import transcriber
from transcriber import TranscriptionError, transcribe_audio


def _write_wav(path: Path, frames: int = 1600) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * frames)


class FakeSegment:
    def __init__(self, text: str):
        self.text = text


class FakeInfo:
    language = "en"
    language_probability = 0.97


class FakeModel:
    def __init__(self, texts=None, error=None):
        self.texts = texts or []
        self.error = error

    def transcribe(self, audio_path, **kwargs):
        if self.error is not None:
            raise self.error
        return iter([FakeSegment(text) for text in self.texts]), FakeInfo()


class TranscriberTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.wav_path = Path(self.temp_dir.name) / "test.wav"
        _write_wav(self.wav_path)
        self.previous_model = transcriber._model
        transcriber._model = None

    def tearDown(self):
        transcriber._model = self.previous_model
        self.temp_dir.cleanup()

    def test_reads_wav_duration(self):
        self.assertAlmostEqual(transcriber._validate_wav(self.wav_path), 0.1, places=2)

    def test_rejects_empty_audio(self):
        empty_path = Path(self.temp_dir.name) / "empty.wav"
        _write_wav(empty_path, frames=0)
        with self.assertRaises(TranscriptionError) as context:
            transcribe_audio(empty_path)
        self.assertEqual(context.exception.category, "empty_audio")

    def test_rejects_missing_audio(self):
        with self.assertRaises(TranscriptionError) as context:
            transcribe_audio(Path(self.temp_dir.name) / "missing.wav")
        self.assertEqual(context.exception.category, "audio_missing")

    def test_joins_segments_and_preserves_punctuation(self):
        transcriber._model = FakeModel(
            texts=[" Hello, world.", " This is a test."],
        )
        result = transcribe_audio(self.wav_path)
        self.assertEqual(result.text, "Hello, world. This is a test.")
        self.assertEqual(result.language, "en")
        self.assertEqual(result.segment_count, 2)
        self.assertAlmostEqual(result.duration_seconds, 0.1, places=2)

    def test_empty_transcription_is_allowed(self):
        transcriber._model = FakeModel(texts=[])
        result = transcribe_audio(self.wav_path)
        self.assertEqual(result.text, "")
        self.assertEqual(result.segment_count, 0)

    def test_transcription_failure_is_wrapped(self):
        transcriber._model = FakeModel(error=RuntimeError("boom"))
        with self.assertLogs("transcriber", level="ERROR"):
            with self.assertRaises(TranscriptionError) as context:
                transcribe_audio(self.wav_path)
        self.assertEqual(context.exception.category, "transcription_failed")


if __name__ == "__main__":
    unittest.main()
