import unittest
from pathlib import Path

from downloader import (
    ReelValidationError,
    _audio_download_command,
    error_from_stderr,
    validate_reel_url,
)


class ValidateReelUrlTests(unittest.TestCase):
    def test_accepts_standard_reel_url(self):
        url, shortcode = validate_reel_url("https://www.instagram.com/reel/DWwuHPfCZ8Z/")
        self.assertEqual(url, "https://www.instagram.com/reel/DWwuHPfCZ8Z/")
        self.assertEqual(shortcode, "DWwuHPfCZ8Z")

    def test_accepts_reels_path_and_query(self):
        url, shortcode = validate_reel_url(
            "http://instagram.com/reels/AbCdE_123-/?igsh=example&hl=en"
        )
        self.assertEqual(url, "https://www.instagram.com/reel/AbCdE_123-/")
        self.assertEqual(shortcode, "AbCdE_123-")

    def test_accepts_mobile_host_without_trailing_slash(self):
        url, shortcode = validate_reel_url("https://m.instagram.com/reel/ABC123")
        self.assertEqual(url, "https://www.instagram.com/reel/ABC123/")
        self.assertEqual(shortcode, "ABC123")

    def test_rejects_missing_or_malformed_urls(self):
        bad_urls = [
            "",
            "not-a-url",
            "ftp://www.instagram.com/reel/ABC123/",
            "file:///etc/passwd",
            "http://127.0.0.1/reel/ABC123/",
            "https://evil.com/reel/ABC123/",
            "https://www.instagram.com.evil.com/reel/ABC123/",
            "https://user:pass@www.instagram.com/reel/ABC123/",
            "https://www.instagram.com:8443/reel/ABC123/",
            "https://www.instagram.com/p/ABC123/",
            "https://www.instagram.com/reel/ABC123/extra",
            "https://www.instagram.com/reel/AB/",
            "https://www.instagram.com/reel/ABC 123/",
            "https://www.instagram.com/reel/ABC123/\nhttps://evil.com/",
        ]
        for bad_url in bad_urls:
            with self.subTest(bad_url=bad_url):
                with self.assertRaises(ReelValidationError):
                    validate_reel_url(bad_url)


class ErrorMappingTests(unittest.TestCase):
    def test_maps_known_errors(self):
        cases = [
            ("ERROR: Instagram sent an empty media response", "unavailable_or_login"),
            ("This content isn't available to everyone", "age_restricted"),
            ("HTTP Error 429: Too Many Requests", "rate_limited"),
            ("ERROR: login required", "login_required"),
            ("This video is private", "private"),
            ("ERROR: Unsupported URL", "unsupported_url"),
            ("Requested format is not available", "no_format"),
            ("File is larger than max-filesize", "too_large"),
            ("ffmpeg not found", "ffmpeg_missing"),
            ("HTTP Error 403: Forbidden", "access_denied"),
            ("HTTP Error 404: Not Found", "unavailable"),
            ("Read timed out", "timeout"),
            ("something unexpected", "unknown"),
        ]
        for stderr, expected_category in cases:
            with self.subTest(stderr=stderr):
                self.assertEqual(error_from_stderr(stderr).category, expected_category)


class AudioDownloadCommandTests(unittest.TestCase):
    def test_forces_ipv4_and_prefers_audio_with_merge_fallback(self):
        url = "https://www.instagram.com/reel/DWwuHPfCZ8Z/"
        temp_dir = Path("/tmp/eureka-audio-test")

        command = _audio_download_command(url, temp_dir)

        self.assertIn("--force-ipv4", command)
        self.assertEqual(command[command.index("-f") + 1], "bestaudio/bv*+ba/b")
        self.assertEqual(command[command.index("-o") + 1], str(temp_dir / "source.%(ext)s"))
        self.assertEqual(command[-1], url)


if __name__ == "__main__":
    unittest.main()
