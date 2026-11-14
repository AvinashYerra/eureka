# Eureka

**Discover what you hear.**

Paste an Instagram Reel. Find the recommendations hidden inside it.

Eureka is an experimental Streamlit proof of concept that accepts a public Instagram Reel URL, retrieves its audio, transcribes it locally with `faster-whisper`, extracts every content recommendation with OpenRouter, searches YouTube for matching videos, and shows a simple list of discoveries with a single YouTube link each.

This is **not** the full Eureka product. It does not include login, a database, payments, advertising, recommendation feedback learning, or deployment. Recommendation extraction and YouTube search are included.

## What it does

1. Validates and canonicalizes the Instagram Reel URL.
2. Retrieves the Reel's audio with `yt-dlp`, preferring an audio-only stream.
3. Extracts audio from a video stream with FFmpeg only when no audio-only stream is available.
4. Saves a 16 kHz mono PCM WAV at `data/audio/<shortcode>.wav`.
5. Transcribes it with the `faster-whisper` `small` model on CPU with automatic language detection.
6. Saves the transcript at `data/transcripts/<shortcode>.txt`.
7. Sends the transcript to OpenRouter to classify whether it is a recommendation Reel and to extract genre-agnostic content recommendations.
8. Searches YouTube for the top two matches for every recommendation's `search_query`. If the Reel is not a recommendation Reel, YouTube search is skipped and the UI shows a message instead.
9. Saves the classification, extracted recommendations, and search metadata at `data/recommendations/<shortcode>.json`.
10. Deletes the saved audio and transcript once recommendations are saved, keeping only the recommendation JSON.
11. Displays each recommendation's title with a single **Watch on YouTube** link for its best matching video. The transcript, audio, and technical details stay out of the UI.

If the Reel genuinely has no audio track, the app shows a clear message and skips transcription. If audio is present but contains no speech, an empty transcript file is saved and the UI reports that no speech was detected. Audio and transcript files are kept when the Reel is not a recommendation Reel, when no recommendations are found, or when recommendation extraction fails.

## Supported languages

Eureka supports **English and Hindi**. The spoken language is detected automatically by `faster-whisper`; there is no language selector, and no extra configuration is required.

## Genre-agnostic recommendation extraction

The extraction prompt does not use a fixed genre list. A recommendation can be a YouTube video, movie, short film, documentary, TV show, web series, song, music video, performance, programming tutorial, technical course, educational lecture, podcast, interview, talk, conference session, product review, recipe, travel guide, fitness tutorial, DIY video, or any other searchable content.

Before extracting recommendations, the model classifies the transcript with a top-level `is_recommendation_reel` boolean. A Reel is treated as a recommendation Reel only when at least one specific recommended item can be identified. Vlogs, rants, comedy sketches, news recaps, general advice, motivational speeches, and advertisements without specific outside recommendations are classified as `false`; the app then shows a message and skips YouTube search.

For each recommendation the model returns:

- `title`: exact title if spoken, otherwise empty
- `search_query`: concise YouTube search query
- `creator_or_channel`: only when established
- `category`: flexible descriptive category
- `topic`: subject of the content
- `context`: why or how it was recommended
- `confidence`: `high`, `medium`, or `low`
- `title_type`: `exact`, `inferred`, or `descriptive`

Post-processing validates the response before it reaches the UI:

- `exact` titles are preserved only when found in the transcript; otherwise they are downgraded to inferred and confidence is lowered.
- A creator/channel that cannot be found in the transcript is removed rather than trusted.
- `descriptive` recommendations never receive an invented exact title.
- A missing search query falls back to title, creator, and topic.
- Invalid confidence values become `low`; invalid title types are derived from available fields.
- Duplicate recommendations are merged by query and, when safe, by title/creator.
- Output is capped at 12 recommendations to keep the UI and searches bounded.
- Empty transcripts skip the LLM entirely.
- When `is_recommendation_reel` is `false`, recommendations are discarded and YouTube search is skipped.

The prompt instructs the model to distinguish actual recommendations from passing references, examples, unrelated mentions, and general topics, and to never invent a title, creator, channel, or URL. Sponsored recommendations count when the speaker clearly recommends the content.

## OpenRouter integration

- The app sends one Chat Completions request to `https://openrouter.ai/api/v1/chat/completions`.
- The complete transcript is sent in that single request. It is not chunked or truncated.
- The default model is `openrouter/free`; set `OPENROUTER_MODEL` in `.env` or the environment to pin a specific model later.
- The request uses a strict JSON Schema through `response_format`, so compatible models must return the recommendation object shape.
- The response is parsed and normalized again locally. Malformed JSON, a missing `recommendations` array, a missing or non-boolean `is_recommendation_reel` flag, empty output, and truncated responses are rejected with a user-facing warning.
- Rate limits, unavailable models, authentication failures, insufficient credits, timeouts, network failures, and other upstream errors are mapped to friendly messages.
- The request includes only `openrouter/free` by default. It does **not** include a paid fallback array. If you set a paid model explicitly through `OPENROUTER_MODEL` in `.env` or the environment, that is your explicit configuration, not an automatic fallback.
- There is no fallback to another LLM provider or any paid API.
- The OpenRouter transport is isolated in `openrouter_client.py`; `recommendations.py` contains the prompt, schema, validation, deduplication, and a small default completion adapter. Switching providers means replacing that adapter and the transport module.

## YouTube search

YouTube search uses `yt-dlp`'s `ytsearch2:` extractor. No YouTube Data API key is required.

For every extracted recommendation:

- The app searches using `search_query`, not just `title`.
- The top two results are retained and deduplicated by video ID; the first result is the ranked best match.
- The UI shows only the recommendation title and a single `Watch on YouTube` link for the best match.
- Before each search, the cleaned query is printed to the terminal as `YouTube search query: <query>`.
- If a search returns no match, the recommendation is still listed without a link.
- Search failures are handled per recommendation and do not remove other results.
- Titles and video URLs come from YouTube's search metadata; they are not generated by the app.

## Requirements

- Python 3.12 deployment target; tested with Python 3.12.13 and 3.14.7
- FFmpeg and ffprobe on the server `PATH`
- An OpenRouter API key for recommendation extraction; the default `openrouter/free` router uses free models
- `yt-dlp` with its `curl-cffi` impersonation extra, installed by `requirements.txt`
- `pip`
- Disk space for Python packages, the Whisper model, and saved files

A YouTube Data API key is **not** required.

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a key at <https://openrouter.ai/keys>, then create `.env` in the project root:

```bash
cat > .env <<'EOF'
OPENROUTER_API_KEY=your-api-key
OPENROUTER_MODEL=openrouter/free
EOF
```

The app loads `.env` automatically at startup. `.env` is gitignored; do not commit it. Existing process environment variables take precedence over `.env`, so you can also use:

```bash
export OPENROUTER_API_KEY="your-api-key"
```

For Streamlit deployments, place the same values in `.streamlit/secrets.toml` locally or in the app's **Secrets** settings. Copy `.streamlit/secrets.toml.example` to get started. The app reads `st.secrets` first, then `.env`, and process environment values always take precedence. `.streamlit/secrets.toml` is gitignored; never commit real keys.

The default model is `openrouter/free`, OpenRouter's free-model router. To pin a specific model, set its full OpenRouter slug in `.env` or the environment:

```bash
OPENROUTER_MODEL=provider/model:free
```

Install FFmpeg if needed:

```bash
# macOS
brew install ffmpeg

# Debian/Ubuntu
sudo apt-get update && sudo apt-get install -y ffmpeg
```

The `av==15.1.0` pin is intentional. Newer PyAV releases changed the `av.open(..., metadata_errors=...)` API used by `faster-whisper` 1.2.1 and can fail at transcription time.

`yt-dlp` requires its `curl-cffi` extra for browser impersonation when Instagram serves anonymous requests. Installing from `requirements.txt` includes it automatically.

The app reads `OPENROUTER_API_KEY` and optional `OPENROUTER_MODEL` from `st.secrets`, the project-root `.env`, or the backend process environment. Process environment values take precedence, followed by `st.secrets`, then `.env`. There is no key field in the UI, and the key is never included in the request body. Restart Streamlit after changing secrets or `.env`.

## Run

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, paste a public Reel URL, and click **Find recommendations**. A small loading indicator shows the current high-level step:

1. Processing Reel...
2. Understanding recommendations...
3. Finding videos on YouTube...
4. Finishing up...

The result section shows **Your discoveries**, with each recommendation title and a single **Watch on YouTube** link. If nothing was recommended, the app shows `No discoveries found in this Reel.` If the Reel is not a recommendation Reel, the app shows a clear message and skips YouTube search. After recommendations are saved, the audio and transcript files are deleted; the recommendation JSON remains on disk.

## Deploy to Streamlit Community Cloud

1. Push the repository to GitHub. Do not commit `.env`, `.streamlit/secrets.toml`, `data/`, model files, downloaded media, or `.venv/`.
2. In Streamlit Community Cloud, create an app with `app.py` as the entrypoint.
3. In **Advanced settings**, select **Python 3.12**. Community Cloud does not use `runtime.txt` or `.python-version` to choose the runtime.
4. Add the secrets in the app dashboard:

   ```toml
   OPENROUTER_API_KEY = "your-openrouter-api-key"
   OPENROUTER_MODEL = "openrouter/free"
   ```

5. Deploy. `packages.txt` installs FFmpeg, ffprobe, and `libgomp1` via `apt-get`; `requirements.txt` installs the pinned Python dependencies.
6. The first transcription downloads the `faster-whisper` `small` model automatically into the Hugging Face cache. `st.cache_resource` keeps the loaded model in memory across reruns and sessions, so it is not reloaded for every request.

Community Cloud storage is ephemeral. The Whisper cache and `data/` directory can be cleared when the app restarts, and are recreated automatically when needed.

### Deployment troubleshooting

- If Instagram returns an empty media response, a login wall, a rate limit, or an access-denied error, the UI explains that Instagram may be blocking downloads from the cloud server. Try another public Reel or retry later. This is common on shared cloud IPs.
- If transcription fails with a server-tools error, confirm `ffmpeg` and `libgomp1` are installed from `packages.txt` and that the app was rebooted after the file changed.
- If OpenRouter reports missing credentials, confirm `OPENROUTER_API_KEY` is set in the app's Secrets and restart the app.

### Privacy note

- Audio is processed locally and is **not** sent to OpenRouter.
- The transcript text is sent to OpenRouter for recommendation extraction. OpenRouter may route it to a third-party model provider depending on the configured model, availability, and required capabilities.
- Recommendation search queries are sent to YouTube through `yt-dlp`.
- Recommendation metadata is saved locally. Audio and transcripts are deleted locally after recommendations are saved. There is no database or cloud storage.
- Do not use this proof of concept with content or transcripts you are not permitted to send to these services.

## Local files

The app creates these directories automatically:

```text
data/
├── audio/
│   └── <shortcode>.wav
├── transcripts/
│   └── <shortcode>.txt
├── recommendations/
│   └── <shortcode>.json
└── .locks/
    └── <shortcode>.lock
```

- Files use the Reel shortcode as the filename.
- The recommendation JSON is retained locally. Audio and transcript files are deleted once recommendations are saved, and are kept when the Reel is not a recommendation Reel, when no recommendations are found, or when recommendation extraction fails.
- Writes use temporary files plus `os.replace()` for atomic replacement.
- A per-Reel file lock prevents two processes from processing the same shortcode at the same time.
- Temporary download and FFmpeg working files are deleted after processing.
- The recommendation JSON includes the recommendation Reel classification, transcript language, extraction error (if any), each recommendation, its YouTube results, and per-recommendation search errors.

Example shape:

```json
{
  "shortcode": "Dfgc-thSjNN",
  "language": "en",
  "is_recommendation_reel": true,
  "generated_at": "2026-11-14T00:00:00+00:00",
  "recommendations": [
    {
      "title": "Dinner for Few",
      "search_query": "Dinner for Few short film",
      "creator_or_channel": null,
      "category": "Animated short film",
      "topic": "Wealth inequality",
      "context": "The speaker recommends this short film about inequality.",
      "confidence": "medium",
      "title_type": "inferred",
      "youtube_results": [],
      "search_error": null
    }
  ],
  "error": null
}
```

## Error handling

- Invalid or malicious URLs are rejected before any network request.
- Unavailable, private, age-restricted, rate-limited, and login-walled Reels produce friendly messages.
- Missing audio streams are treated as a successful no-audio result, with no files written.
- FFmpeg failures are mapped to a clear media-processing error.
- Empty audio and empty transcripts are handled separately.
- Transcription failures are reported without displaying raw exceptions or tracebacks.
- A missing OpenRouter key, authentication failure, rate limit, unavailable model, timeout, blocked response, malformed JSON, or other invalid LLM response does not delete the audio or transcript. The UI shows a warning and continues to display the successful audio/transcript result.
- A Reel classified as not a recommendation Reel shows a clear message and skips YouTube search.
- Recommendation extraction does not fall back to another LLM provider or paid API.
- If recommendation JSON cannot be saved, the extracted cards remain visible in the current session and a warning is shown.
- A YouTube search failure affects only that recommendation; other recommendations and their results remain available.
- Technical details such as error categories, return codes, and sanitized stderr are written to the server log. Raw API errors, Instagram signed media URLs, and transcript content are not logged.

## Security and resource controls

- Strict Instagram hostname and path allowlist, preventing arbitrary URL fetching and SSRF.
- Rejects credentials, non-standard ports, non-HTTP schemes, and non-Reel paths.
- 100 MB download limit and 5-minute audio duration limit.
- 120-second download timeout; 60-second FFmpeg timeout; 20-second Instagram socket timeout.
- 30-second timeout per YouTube search; 90-second timeout per OpenRouter request.
- One active processing job per Streamlit process, plus a per-Reel cross-process lock.
- The OpenRouter API key is read from the backend `.env` or process environment only and is never rendered or included in the request body. `.env` is gitignored.
- Recommendation extraction has no automatic retry or fallback to another provider.
- OpenRouter output is schema-constrained and validated again locally. The request asks for a strict JSON schema and the app re-validates the parsed response.
- Descriptive recommendations cannot acquire invented exact titles.
- YouTube URLs are taken from `yt-dlp` metadata or canonically constructed from a returned video ID.
- No raw exceptions, absolute server paths, Instagram signed media URLs, or API keys are exposed in the UI.
- No cookies, Instagram accounts, or authentication bypasses.

## Testing

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Current result: **96 tests pass** with Python 3.12.13. Coverage includes URL validation, error mapping, audio-source selection, FFmpeg conversion, atomic storage, cleanup, locking, transcription behavior, recommendation Reel classification, OpenRouter request shape, `.env` loading, Streamlit secrets loading, HTTP error mapping, recommendation prompt/schema handling, malformed JSON, normalization and deduplication, YouTube result parsing and two-result limits, and pipeline orchestration. Audio/FFmpeg tests are skipped automatically if FFmpeg is unavailable.

Run the live audio/transcription/recommendation flow manually:

```bash
python - <<'PY'
from pipeline import process_reel

result = process_reel(
    "https://www.instagram.com/reel/Dfgc-thSjNN/",
    on_stage=print,
)
print("Language:", result.language)
print("Transcript:", result.transcript)
print("Recommendations:", result.recommendations)
print("Recommendation error:", result.recommendation_error)
PY
```

Live results from 2026-11-13 and 2026-11-14 on a residential IP:

| Case | Result |
|---|---|
| `CteWys7p45N` | Audio-only stream, 59.0 s, 1.80 MB WAV, English transcript, 14 segments |
| `Dfgc-thSjNN` | 40.2 s, 717-character English transcript about short films to watch on YouTube |
| `Db3Ej0NCYbF` | Audio-only stream, 56.5 s, English transcript |
| `DJuBcywMHts` | Audio-only stream, 32.9 s, no speech detected; empty transcript |
| `Crdons0AVeh` | No audio stream; transcription skipped, no files written |
| `DIPIf1UTqLg` | Unavailable/login error handled |
| `DagG8upSTlQ` | Age-restricted error handled |

Live YouTube search checks on 2026-11-14:

| Query | Result |
|---|---|
| `Alex The Analyst beginner SQL tutorial` | 2 matches, first result on the mentioned channel |
| `database indexes explained tutorial` | 2 matches, varied channels and durations |

These results are point-in-time and depend on Instagram's anonymous access behavior, OpenRouter availability, and YouTube search results for the current IP.

## Troubleshooting

**Recommendations are unavailable and the app says `OPENROUTER_API_KEY` is missing**  
Add the key to the project-root `.env` (or export it) and restart Streamlit:

```bash
echo 'OPENROUTER_API_KEY=your-api-key' >> .env
streamlit run app.py
```

**OpenRouter returns a model-unavailable or not-found error**  
Free-model availability changes frequently. Retry, keep the router at `openrouter/free`, or pin another free model:

```bash
# In .env:
OPENROUTER_MODEL=provider/model:free
```

Restart Streamlit after changing `.env`.

**OpenRouter returns `429` or rate-limit errors**  
Free models have shared rate limits. Wait and retry; if the problem persists, check the limits on your OpenRouter account. Audio and transcripts remain saved.

**OpenRouter returns malformed JSON or a schema error**  
The app rejects the response and shows a warning instead of guessing. Retry. If it persists, pin a model that supports structured outputs or use `openrouter/free`, which routes only to free models that support the requested features. Audio and transcripts remain saved.

**OpenRouter reports insufficient credits**  
This should not happen with `openrouter/free`. If you explicitly configured a paid `OPENROUTER_MODEL`, switch back to `openrouter/free` or add credits.

**YouTube search fails**  
Confirm `yt-dlp` is installed in the active environment and that the network can reach YouTube:

```bash
python -m yt_dlp --version
python -m yt_dlp --dump-single-json --flat-playlist "ytsearch2:test query" >/dev/null
```

Search failures affect only the affected recommendation.

**`TypeError: open() got an unexpected keyword argument 'metadata_errors'`**  
`faster-whisper` 1.2.1 is incompatible with newer PyAV releases. Reinstall the pinned version:

```bash
pip install "av==15.1.0"
```

**First transcription is slow**  
The first run downloads and loads the `small` model (about 500 MB). Later runs reuse the local Hugging Face cache. CPU transcription can still take several seconds per minute of audio.

**Instagram returns an empty media response or login challenge**  
First confirm the virtual environment has the impersonation dependency:

```bash
pip install -r requirements.txt
python -c "import curl_cffi; print(curl_cffi.__version__)"
```

Anonymous access is still IP-dependent. A local residential connection may work while a cloud/datacenter IP fails. This app intentionally does not add cookies or account workarounds.

**FFmpeg is missing**  
Install FFmpeg and confirm both commands work:

```bash
ffmpeg -version
ffprobe -version
```

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, recommendation cards, YouTube result cards, processing-stage display |
| `downloader.py` | URL validation, audio retrieval, FFmpeg conversion, media validation |
| `transcriber.py` | `faster-whisper` model loading and transcription |
| `openrouter_client.py` | OpenRouter Chat Completions transport, API key/model configuration, structured-output request, HTTP error mapping |
| `recommendations.py` | Recommendation prompt, JSON schema, validation, normalization, deduplication, default completion adapter |
| `youtube_search.py` | `yt-dlp` YouTube search and metadata parsing |
| `storage.py` | Local directories, per-Reel lock, atomic audio/transcript/recommendation writes |
| `pipeline.py` | End-to-end audio, transcription, recommendation, and search orchestration |
| `.streamlit/config.toml` | Dark theme, amber accent, and local static asset serving |
| `static/desert-sunset.jpg` | Full-screen New Mexico desert background photograph |
| `static/oswald-latin.woff2` | Locally hosted condensed display typeface |
| `static/favicon.png` | Eureka favicon |
| `requirements.txt` | Python dependencies, including the PyAV compatibility pin |
| `packages.txt` | System dependency for FFmpeg on supported hosts |
| `tests/test_downloader.py` | URL validation and error mapping tests |
| `tests/test_audio_downloader.py` | Audio-source probing, preference, and conversion tests |
| `tests/test_storage.py` | Atomic storage, recommendation JSON, and locking tests |
| `tests/test_transcriber.py` | Transcription behavior and empty-audio tests |
| `tests/test_recommendations.py` | Prompt/schema handling, response normalization, deduplication, and malformed-response tests |
| `tests/test_openrouter_client.py` | OpenRouter request shape, key/model configuration, and HTTP error-mapping tests |
| `tests/test_youtube_search.py` | YouTube search parsing, two-result limits, query output, and error tests |
| `tests/test_pipeline.py` | End-to-end orchestration with mocked media, LLM, and search |

## Brand assets and credits

- Background photograph: **Cabezon Peak sunset and Moon, New Mexico** by BrandonStephenson, released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) via Wikimedia Commons.
- Display typeface: **Oswald** by Vernon Adams, released under the [SIL Open Font License 1.1](https://openfontlicense.org/).
- The interface uses the atmosphere of the New Mexico high desert. It does not include any third-party television logos, characters, actors, or promotional artwork.

## Disclaimer

This is an experimental technical proof of concept. Instagram does not officially authorize arbitrary Reel downloading through this method, and automated collection may violate its terms. Transcripts are sent to OpenRouter, which may route them to third-party model providers, and search queries are sent to YouTube. Review OpenRouter's, the selected model provider's, and YouTube's terms and privacy policies. Recommendation extraction can be wrong, and YouTube search results are only candidates, not verified matches. Always let a human choose the correct video. Use only content you have the right to access, download, and process, and obtain legal review before any public deployment.
