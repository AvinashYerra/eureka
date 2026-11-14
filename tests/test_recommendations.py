import json
import os
import unittest
from unittest import mock

import recommendations
from openrouter_client import OpenRouterError
from recommendations import (
    RecommendationError,
    RecommendationExtraction,
    extract_recommendations,
)


class FakeCompletion:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = 0
        self.messages = None

    def __call__(self, messages):
        self.calls += 1
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.content


def _payload(recommendations, is_recommendation_reel=True):
    return json.dumps(
        {
            "is_recommendation_reel": is_recommendation_reel,
            "recommendations": recommendations,
        }
    )


class RecommendationTests(unittest.TestCase):
    def test_empty_transcript_returns_no_recommendations(self):
        completion = FakeCompletion(error=AssertionError("completion should not be used"))
        self.assertEqual(
            extract_recommendations("", complete=completion),
            RecommendationExtraction(False, ()),
        )
        self.assertEqual(completion.calls, 0)

    def test_extracts_and_normalizes_multiple_recommendations(self):
        payload = {
            "recommendations": [
                {
                    "title": "The Exact Video",
                    "search_query": "",
                    "creator_or_channel": "Example Channel",
                    "category": "Technical tutorial",
                    "topic": "Python",
                    "context": "Recommended for learning Python.",
                    "confidence": "high",
                    "title_type": "exact",
                },
                {
                    "title": "Invented descriptive title",
                    "search_query": "database indexes explained tutorial",
                    "creator_or_channel": "",
                    "category": "Technical educational video",
                    "topic": "Database indexing",
                    "context": "A video explaining database indexes.",
                    "confidence": "medium",
                    "title_type": "descriptive",
                },
                {
                    "title": "",
                    "search_query": "a video with no known title",
                    "creator_or_channel": "",
                    "category": "Other",
                    "topic": "Something",
                    "context": "",
                    "confidence": "unexpected",
                    "title_type": "exact",
                },
                {
                    "title": "Partial title",
                    "search_query": "The Exact Video Example Channel Python",
                    "creator_or_channel": "Example Channel",
                    "category": "Technical tutorial",
                    "topic": "Python",
                    "context": "Recommended again.",
                    "confidence": "medium",
                    "title_type": "inferred",
                },
            ]
        }
        completion = FakeCompletion(content=_payload(payload["recommendations"]))

        extraction = extract_recommendations(
            "A transcript about The Exact Video from Example Channel, covering Python.",
            language="en",
            complete=completion,
        )
        result = extraction.recommendations

        self.assertTrue(extraction.is_recommendation_reel)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].title, "The Exact Video")
        self.assertEqual(
            result[0].search_query,
            "The Exact Video Example Channel Python",
        )
        self.assertEqual(result[0].confidence, "high")
        self.assertEqual(result[0].title_type, "exact")
        self.assertIsNone(result[1].title)
        self.assertEqual(result[1].title_type, "descriptive")
        self.assertIsNone(result[2].title)
        self.assertEqual(result[2].title_type, "descriptive")
        self.assertEqual(result[2].confidence, "low")
        self.assertEqual(completion.calls, 1)

    def test_deduplicates_by_search_query_and_keeps_higher_ranked_item(self):
        payload = {
            "recommendations": [
                {
                    "title": "A title",
                    "search_query": "Same Query",
                    "creator_or_channel": "",
                    "category": "One",
                    "topic": "One",
                    "context": "First.",
                    "confidence": "low",
                    "title_type": "descriptive",
                },
                {
                    "title": "A better title",
                    "search_query": "same query",
                    "creator_or_channel": "",
                    "category": "Two",
                    "topic": "Two",
                    "context": "Second.",
                    "confidence": "high",
                    "title_type": "inferred",
                },
            ]
        }
        completion = FakeCompletion(content=_payload(payload["recommendations"]))

        result = extract_recommendations("A transcript", complete=completion).recommendations

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].title, "A better title")
        self.assertEqual(result[0].confidence, "high")

    def test_deduplicates_same_title_and_creator_with_different_queries(self):
        payload = {
            "recommendations": [
                {
                    "title": "The Same Video",
                    "search_query": "creator the same video",
                    "creator_or_channel": "Creator",
                    "category": "One",
                    "topic": "One",
                    "context": "First.",
                    "confidence": "low",
                    "title_type": "inferred",
                },
                {
                    "title": "The Same Video",
                    "search_query": "a different query",
                    "creator_or_channel": "Creator",
                    "category": "Two",
                    "topic": "Two",
                    "context": "Second.",
                    "confidence": "high",
                    "title_type": "exact",
                },
            ]
        }
        completion = FakeCompletion(content=_payload(payload["recommendations"]))

        result = extract_recommendations("A transcript", complete=completion).recommendations

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].search_query, "a different query")

    def test_does_not_merge_short_unattributed_titles(self):
        payload = {
            "recommendations": [
                {
                    "title": "Intro",
                    "search_query": "first intro",
                    "creator_or_channel": "",
                    "category": "One",
                    "topic": "One",
                    "context": "First.",
                    "confidence": "medium",
                    "title_type": "inferred",
                },
                {
                    "title": "Intro",
                    "search_query": "second intro",
                    "creator_or_channel": "",
                    "category": "Two",
                    "topic": "Two",
                    "context": "Second.",
                    "confidence": "medium",
                    "title_type": "inferred",
                },
            ]
        }
        completion = FakeCompletion(content=_payload(payload["recommendations"]))

        result = extract_recommendations("A transcript", complete=completion).recommendations

        self.assertEqual(len(result), 2)

    def test_unverified_exact_title_is_downgraded(self):
        payload = {
            "recommendations": [
                {
                    "title": "A Title Not In The Transcript",
                    "search_query": "a title not in the transcript",
                    "creator_or_channel": "Invented Creator",
                    "category": "Other",
                    "topic": "Something",
                    "context": "Recommended.",
                    "confidence": "high",
                    "title_type": "exact",
                }
            ]
        }
        completion = FakeCompletion(content=_payload(payload["recommendations"]))

        result = extract_recommendations(
            "A completely different transcript.",
            complete=completion,
        ).recommendations

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].title_type, "inferred")
        self.assertEqual(result[0].confidence, "medium")
        self.assertIsNone(result[0].creator_or_channel)

    def test_preserves_non_ascii_titles_and_creators(self):
        payload = {
            "recommendations": [
                {
                    "title": "tutorial de programación",
                    "search_query": "Ana tutorial de programación",
                    "creator_or_channel": "Ana",
                    "category": "Tutorial",
                    "topic": "Programación",
                    "context": "Recomendado para aprender.",
                    "confidence": "high",
                    "title_type": "exact",
                }
            ]
        }
        completion = FakeCompletion(content=_payload(payload["recommendations"]))

        result = extract_recommendations(
            "Mira el tutorial de programación de Ana en español.",
            complete=completion,
        ).recommendations

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].title, "tutorial de programación")
        self.assertEqual(result[0].creator_or_channel, "Ana")
        self.assertEqual(result[0].title_type, "exact")

    def test_preserves_devanagari_title_and_creator(self):
        payload = {
            "recommendations": [
                {
                    "title": "बाहुबली",
                    "search_query": "Baahubali SS Rajamouli",
                    "creator_or_channel": "एस. एस. राजमौली",
                    "category": "फिल्म",
                    "topic": "बाहुबली",
                    "context": "एस. एस. राजमौली द्वारा निर्देशित फिल्म के रूप में सुझाया गया।",
                    "confidence": "high",
                    "title_type": "exact",
                }
            ]
        }
        completion = FakeCompletion(content=_payload(payload["recommendations"]))

        result = extract_recommendations(
            "आज मैं आपको बाहुबली फिल्म देखने की सलाह दूंगा, जिसे एस. एस. राजमौली ने निर्देशित किया है।",
            language="hi",
            complete=completion,
        ).recommendations

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].title, "बाहुबली")
        self.assertEqual(result[0].creator_or_channel, "एस. एस. राजमौली")
        self.assertEqual(result[0].category, "फिल्म")
        self.assertEqual(result[0].search_query, "Baahubali SS Rajamouli")

    def test_appends_creator_to_native_script_search_query(self):
        payload = {
            "recommendations": [
                {
                    "title": "बाहुबली",
                    "search_query": "बाहुबली फिल्म",
                    "creator_or_channel": "एस. एस. राजमौली",
                    "category": "फिल्म",
                    "topic": "बाहुबली",
                    "context": "एस. एस. राजमौली द्वारा निर्देशित फिल्म के रूप में सुझाया गया।",
                    "confidence": "high",
                    "title_type": "exact",
                }
            ]
        }
        completion = FakeCompletion(content=_payload(payload["recommendations"]))

        result = extract_recommendations(
            "आज मैं आपको बाहुबली फिल्म देखने की सलाह दूंगा, जिसे एस. एस. राजमौली ने निर्देशित किया है।",
            language="hi",
            complete=completion,
        ).recommendations

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0].search_query,
            "बाहुबली फिल्म एस. एस. राजमौली",
        )

    def test_system_prompt_requires_transcript_language_and_creator_preservation(self):
        completion = FakeCompletion(content=_payload([]))
        extract_recommendations("A transcript", complete=completion)

        system_message = completion.messages[0]["content"]
        self.assertIn("LANGUAGE AND SCRIPT", system_message)
        self.assertIn("never translate the whole transcript", system_message)
        self.assertIn("language and script used in the transcript", system_message)
        self.assertIn("include them in search_query", system_message)
        self.assertIn("Never use English category labels", system_message)
        self.assertIn("FINAL LANGUAGE CHECK", system_message)

    def test_parses_markdown_wrapped_json(self):
        completion = FakeCompletion(
            content=f"```json\n{_payload([])}\n```",
        )
        self.assertEqual(
            extract_recommendations("A transcript", complete=completion),
            RecommendationExtraction(True, ()),
        )

    def test_parses_json_with_surrounding_text(self):
        completion = FakeCompletion(
            content=f"Here is the result:\n{_payload([])}\nDone.",
        )
        self.assertEqual(
            extract_recommendations("A transcript", complete=completion),
            RecommendationExtraction(True, ()),
        )

    def test_non_recommendation_reel_returns_empty_without_normalizing(self):
        payload = {
            "recommendations": [
                {
                    "title": "Should be ignored",
                    "search_query": "should be ignored",
                    "creator_or_channel": "",
                    "category": "One",
                    "topic": "One",
                    "context": "First.",
                    "confidence": "high",
                    "title_type": "exact",
                }
            ]
        }
        completion = FakeCompletion(
            content=_payload(payload["recommendations"], is_recommendation_reel=False)
        )

        extraction = extract_recommendations("A personal vlog", complete=completion)

        self.assertFalse(extraction.is_recommendation_reel)
        self.assertEqual(extraction.recommendations, ())

    def test_missing_recommendation_reel_flag_is_reported(self):
        completion = FakeCompletion(content=json.dumps({"recommendations": []}))
        with self.assertRaises(RecommendationError) as context:
            extract_recommendations("A transcript", complete=completion)
        self.assertEqual(context.exception.category, "invalid_response")

    def test_invalid_json_is_reported(self):
        completion = FakeCompletion(content="not json")
        with self.assertRaises(RecommendationError) as context:
            extract_recommendations("A transcript", complete=completion)
        self.assertEqual(context.exception.category, "invalid_response")

    def test_malformed_schema_is_reported(self):
        completion = FakeCompletion(content='{"recommendations": "not a list"}')
        with self.assertRaises(RecommendationError) as context:
            extract_recommendations("A transcript", complete=completion)
        self.assertEqual(context.exception.category, "invalid_response")

    def test_empty_response_is_reported(self):
        completion = FakeCompletion(content="")
        with self.assertRaises(RecommendationError) as context:
            extract_recommendations("A transcript", complete=completion)
        self.assertEqual(context.exception.category, "invalid_response")

    def test_provider_error_is_mapped(self):
        completion = FakeCompletion(
            error=OpenRouterError("OpenRouter is rate-limiting requests.", "rate_limited")
        )
        with self.assertRaises(RecommendationError) as context:
            extract_recommendations("A transcript", complete=completion)
        self.assertEqual(context.exception.category, "rate_limited")
        self.assertEqual(
            context.exception.user_message,
            "OpenRouter is rate-limiting requests.",
        )

    def test_missing_credentials_are_reported(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RecommendationError) as context:
                extract_recommendations("A transcript")
        self.assertEqual(context.exception.category, "missing_credentials")
        self.assertIn("OPENROUTER_API_KEY", context.exception.user_message)

    def test_complete_transcript_is_sent_in_one_request(self):
        long_transcript = "A" * 150_000
        completion = FakeCompletion(content=_payload([]))

        extract_recommendations(long_transcript, complete=completion)

        self.assertEqual(completion.calls, 1)
        user_message = completion.messages[1]["content"]
        self.assertIn(long_transcript, user_message)

    def test_system_prompt_contains_json_schema(self):
        completion = FakeCompletion(content=_payload([]))
        extract_recommendations("A transcript", complete=completion)

        system_message = completion.messages[0]["content"]
        self.assertIn("JSON schema:", system_message)
        self.assertIn('"is_recommendation_reel"', system_message)
        self.assertIn('"recommendations"', system_message)
        self.assertIn('"title_type"', system_message)

    def test_system_prompt_requires_recommendation_reel_classification(self):
        completion = FakeCompletion(content=_payload([]))
        extract_recommendations("A transcript", complete=completion)

        system_message = completion.messages[0]["content"]
        self.assertIn("RECOMMENDATION REEL", system_message)
        self.assertIn("is_recommendation_reel to false", system_message)
        self.assertIn("return an empty recommendations array", system_message)

    def test_default_completion_uses_openrouter_adapter(self):
        with mock.patch(
            "recommendations.chat_completion",
            return_value=_payload([]),
        ) as completion:
            extract_recommendations("A transcript")

        messages = completion.call_args.args[0]
        self.assertEqual(len(messages), 2)
        self.assertEqual(completion.call_args.kwargs["schema_name"], "reel_recommendations")
        self.assertIs(
            completion.call_args.kwargs["response_schema"],
            recommendations.RESPONSE_SCHEMA,
        )

    def test_extracts_from_mocked_openrouter_response(self):
        recommendation = {
            "title": "The Exact Video",
            "search_query": "The Exact Video Example Channel",
            "creator_or_channel": "Example Channel",
            "category": "Tutorial",
            "topic": "Testing",
            "context": "Recommended for testing.",
            "confidence": "high",
            "title_type": "exact",
        }
        envelope = {
            "model": "test/free",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": _payload([recommendation]),
                    },
                }
            ],
        }
        with mock.patch(
            "openrouter_client._post_json",
            return_value=(200, json.dumps(envelope).encode("utf-8")),
        ):
            with mock.patch.dict(
                os.environ,
                {"OPENROUTER_API_KEY": "test-key"},
                clear=True,
            ):
                extraction = extract_recommendations(
                    "Watch The Exact Video from Example Channel for testing."
                )

        result = extraction.recommendations
        self.assertTrue(extraction.is_recommendation_reel)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].title, "The Exact Video")
        self.assertEqual(result[0].creator_or_channel, "Example Channel")


if __name__ == "__main__":
    unittest.main()
