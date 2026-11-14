import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from downloader import (
    NoAudioError,
    _convert_audio_to_wav,
    _probe_source,
    _select_audio_source,
    _wav_duration,
)

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _write_wav(path: Path, frames: int = 1600) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * frames)


def _make_video(path: Path, with_audio: bool) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=160x120:d=0.5",
    ]
    if with_audio:
        command.extend(
            [
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=0.5",
                "-shortest",
                "-c:a",
                "aac",
            ]
        )
    command.extend(["-c:v", "mpeg4", str(path)])
    subprocess.run(command, check=True, timeout=30)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg and ffprobe are required")
class AudioSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_probes_audio_only_source(self):
        audio_path = self.root / "audio.wav"
        _write_wav(audio_path)
        info = _probe_source(audio_path)
        self.assertTrue(info["has_audio"])
        self.assertFalse(info["has_video"])
        self.assertGreater(info["duration"], 0)

    def test_probes_video_only_source(self):
        video_path = self.root / "video.mp4"
        _make_video(video_path, with_audio=False)
        info = _probe_source(video_path)
        self.assertTrue(info["has_video"])
        self.assertFalse(info["has_audio"])

    def test_prefers_audio_only_source(self):
        audio_path = self.root / "audio.wav"
        video_path = self.root / "video.mp4"
        _write_wav(audio_path)
        _make_video(video_path, with_audio=False)

        selected, info = _select_audio_source(self.root)

        self.assertEqual(selected, audio_path)
        self.assertFalse(info["has_video"])

    def test_rejects_video_without_audio(self):
        video_path = self.root / "video.mp4"
        _make_video(video_path, with_audio=False)
        with self.assertRaises(NoAudioError):
            _select_audio_source(self.root)

    def test_converts_video_audio_to_whisper_wav(self):
        video_path = self.root / "video_audio.mp4"
        wav_path = self.root / "converted.wav"
        _make_video(video_path, with_audio=True)

        _convert_audio_to_wav(video_path, wav_path)
        duration = _wav_duration(wav_path)

        self.assertAlmostEqual(duration, 0.5, places=1)
        with wave.open(str(wav_path), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getframerate(), 16000)
            self.assertEqual(wav_file.getsampwidth(), 2)


if __name__ == "__main__":
    unittest.main()
