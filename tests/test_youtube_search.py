import contextlib
import io
import json
import subprocess
import unittest

from youtube_search import YouTubeSearchError, search_youtube


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        args=["yt-dlp"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class YouTubeSearchTests(unittest.TestCase):
    def test_parses_top_two_and_canonicalizes_urls(self):
        entries = [
            {
                "id": "abcdefghijk",
                "title": "First video",
                "channel": "First channel",
                "duration": 125,
                "thumbnails": [
                    {"url": "https://img.example/small.jpg", "width": 120, "height": 90},
                    {"url": "https://img.example/large.jpg", "width": 1280, "height": 720},
                ],
            },
            {
                "id": "lmnopqrstuv",
                "title": "Second video",
                "uploader": "Second channel",
                "url": "https://www.youtube.com/watch?v=lmnopqrstuv",
                "thumbnail": "https://img.example/second.jpg",
            },
            {
                "id": "abcdefghijk",
                "title": "Duplicate",
                "channel": "Duplicate channel",
            },
            {"id": "short", "title": "Invalid ID"},
            {"id": "wxyzABCDEFG", "title": ""},
            {"id": "zyxwvutsrqp", "title": "Third video"},
            {"id": "qponmlkjihg", "title": "Fourth video"},
            {"id": "fedcbaZYXWV", "title": "Fifth video"},
            {"id": "UTSRQPONMLK", "title": "Sixth video"},
        ]
        captured = {}

        def runner(command, timeout):
            captured["command"] = command
            captured["timeout"] = timeout
            return _completed(json.dumps({"entries": entries}))

        result = search_youtube("test query", runner=runner)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].video_id, "abcdefghijk")
        self.assertEqual(result[0].title, "First video")
        self.assertEqual(result[0].channel, "First channel")
        self.assertEqual(result[0].duration_seconds, 125)
        self.assertEqual(result[0].thumbnail_url, "https://img.example/large.jpg")
        self.assertEqual(
            result[0].url,
            "https://www.youtube.com/watch?v=abcdefghijk",
        )
        self.assertEqual(result[1].url, "https://www.youtube.com/watch?v=lmnopqrstuv")
        self.assertIn("ytsearch2:test query", captured["command"])
        self.assertEqual(captured["timeout"], 30)

    def test_caps_limit_at_two(self):
        entries = [
            {"id": f"video{i:06d}", "title": f"Video {i}"}
            for i in range(8)
        ]

        def runner(command, timeout):
            self.assertIn("ytsearch2:", command[-1])
            return _completed(json.dumps({"entries": entries}))

        self.assertEqual(len(search_youtube("query", limit=10, runner=runner)), 2)

    def test_prints_search_query_before_searching(self):
        captured = {}

        def runner(command, timeout):
            captured["output"] = output.getvalue()
            return _completed(stdout=json.dumps({"entries": []}))

        with contextlib.redirect_stdout(io.StringIO()) as output:
            search_youtube("  test   query  ", runner=runner)

        self.assertIn("YouTube search query: test query", captured["output"])
        self.assertIn("YouTube search query: test query", output.getvalue())

    def test_empty_query_is_rejected(self):
        with self.assertRaises(YouTubeSearchError) as context:
            search_youtube("   ")
        self.assertEqual(context.exception.category, "empty_query")

    def test_nonzero_exit_is_mapped(self):
        def runner(command, timeout):
            return _completed(stderr="ERROR: something failed", returncode=1)

        with self.assertRaises(YouTubeSearchError) as context:
            search_youtube("query", runner=runner)
        self.assertEqual(context.exception.category, "search_failed")

    def test_missing_ytdlp_is_mapped(self):
        def runner(command, timeout):
            return _completed(stderr="No module named yt_dlp", returncode=1)

        with self.assertRaises(YouTubeSearchError) as context:
            search_youtube("query", runner=runner)
        self.assertEqual(context.exception.category, "missing_dependency")

    def test_timeout_is_mapped(self):
        def runner(command, timeout):
            raise subprocess.TimeoutExpired(command, timeout)

        with self.assertRaises(YouTubeSearchError) as context:
            search_youtube("query", runner=runner)
        self.assertEqual(context.exception.category, "timeout")

    def test_invalid_json_is_mapped(self):
        def runner(command, timeout):
            return _completed(stdout="not json")

        with self.assertRaises(YouTubeSearchError) as context:
            search_youtube("query", runner=runner)
        self.assertEqual(context.exception.category, "invalid_response")

    def test_empty_output_returns_no_results(self):
        def runner(command, timeout):
            return _completed(stdout="")

        self.assertEqual(search_youtube("query", runner=runner), ())

    def test_empty_entries_return_no_results(self):
        def runner(command, timeout):
            return _completed(stdout=json.dumps({"entries": []}))

        self.assertEqual(search_youtube("query", runner=runner), ())


if __name__ == "__main__":
    unittest.main()
