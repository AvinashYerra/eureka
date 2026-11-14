from __future__ import annotations

import html
import logging
import threading
from pathlib import Path
from typing import Any

import streamlit as st

from downloader import ReelDownloadError, ReelValidationError
from openrouter_client import load_backend_env
from pipeline import ProcessedReel, RecommendationMatch, process_reel
from storage import StorageError
from transcriber import TranscriptionError

logger = logging.getLogger(__name__)

load_backend_env()

PROCESSING_MESSAGE = "Processing Reel..."
UNDERSTANDING_MESSAGE = "Understanding recommendations..."
YOUTUBE_MESSAGE = "Finding videos on YouTube..."
FINISHING_MESSAGE = "Finishing up..."

FAVICON_PATH = Path(__file__).resolve().parent / "static" / "favicon.png"

STYLES = """
<style>
  @font-face {
    font-family: "Oswald";
    font-style: normal;
    font-weight: 300 700;
    font-display: swap;
    src: url("app/static/oswald-latin.woff2") format("woff2");
  }

  :root {
    --eureka-charcoal: #191B18;
    --eureka-olive: #34382C;
    --eureka-sand: #D8C39A;
    --eureka-mustard: #B49A4B;
    --eureka-amber: #D59B45;
    --eureka-turquoise: #6C9991;
    --eureka-cream: #F1E7D3;
  }

  html, body, .stApp, .stApp button, .stApp input, .stApp textarea {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      "Helvetica Neue", Arial, sans-serif !important;
  }

  .stApp {
    background-color: var(--eureka-charcoal);
    background-image:
      url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160' viewBox='0 0 160 160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='160' height='160' filter='url(%23n)' opacity='0.10'/%3E%3C/svg%3E"),
      radial-gradient(
        ellipse at 50% 36%,
        rgba(25, 27, 24, 0.58) 0%,
        rgba(25, 27, 24, 0.22) 50%,
        rgba(25, 27, 24, 0) 76%
      ),
      radial-gradient(
        ellipse at 50% 30%,
        rgba(213, 155, 69, 0.18) 0%,
        rgba(213, 155, 69, 0) 55%
      ),
      linear-gradient(
        180deg,
        rgba(25, 27, 24, 0.64) 0%,
        rgba(25, 27, 24, 0.2) 28%,
        rgba(25, 27, 24, 0.44) 62%,
        rgba(25, 27, 24, 0.94) 100%
      ),
      url("app/static/desert-sunset.jpg");
    background-attachment: fixed;
    background-position:
      0 0,
      center,
      center,
      center,
      center 62%;
    background-repeat: repeat, no-repeat, no-repeat, no-repeat, no-repeat;
    background-size:
      160px 160px,
      cover,
      cover,
      cover,
      cover;
  }

  header[data-testid="stHeader"],
  #MainMenu,
  footer,
  div[data-testid="stToolbar"],
  div[data-testid="stDecoration"],
  div[data-testid="stStatusWidget"] {
    display: none !important;
  }

  .block-container {
    max-width: 680px;
    padding: 10vh 28px 112px;
  }

  .eureka-brand {
    margin: 0 0 54px;
    text-align: center;
  }

  .eureka-brand-row {
    align-items: center;
    display: flex;
    gap: 12px;
    justify-content: center;
  }

  .eureka-mark {
    display: block;
    flex: 0 0 auto;
    height: 26px;
    width: 40px;
  }

  .eureka-wordmark {
    color: var(--eureka-sand);
    font-family: "Oswald", "Arial Narrow", "Helvetica Neue", sans-serif !important;
    font-size: 1.3rem;
    font-weight: 600;
    letter-spacing: 0.34em;
    line-height: 1;
    margin-right: -0.34em;
    text-transform: uppercase;
  }

  .stApp .eureka-tagline {
    color: rgba(216, 195, 154, 0.62);
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.26em;
    line-height: 1.5;
    margin: 13px 0 0;
    text-transform: uppercase;
  }

  .eureka-hero {
    margin: 0 0 34px;
    text-align: center;
  }

  .eureka-hero h1 {
    color: var(--eureka-cream);
    font-family: "Oswald", "Arial Narrow", "Helvetica Neue", sans-serif !important;
    font-size: clamp(2.45rem, 6vw, 3.6rem);
    font-weight: 600;
    letter-spacing: 0.01em;
    line-height: 1.04;
    margin: 0 0 16px;
    text-shadow: 0 2px 28px rgba(0, 0, 0, 0.38);
  }

  .stApp .eureka-description {
    color: rgba(216, 195, 154, 0.78);
    font-size: 1rem;
    line-height: 1.65;
    margin: 0 auto;
    max-width: 46ch;
    text-align: center;
  }

  .stApp .eureka-disclaimer {
    background: rgba(25, 27, 24, 0.38);
    border: 1px solid rgba(216, 195, 154, 0.12);
    border-radius: 4px;
    color: rgba(216, 195, 154, 0.62);
    font-size: 0.78rem;
    line-height: 1.55;
    margin: 16px auto 0;
    max-width: 56ch;
    padding: 10px 16px;
    text-align: center;
  }

  div[data-testid="stForm"] {
    background: rgba(25, 27, 24, 0.56);
    border: 1px solid rgba(216, 195, 154, 0.16);
    border-radius: 6px;
    box-shadow: 0 24px 64px rgba(0, 0, 0, 0.28);
    padding: 20px;
    -webkit-backdrop-filter: blur(10px);
    backdrop-filter: blur(10px);
  }

  div[data-testid="stTextInput"] div[data-baseweb="input"] {
    background-color: rgba(52, 56, 44, 0.72);
    border: 1px solid rgba(216, 195, 154, 0.22);
    border-radius: 4px;
    min-height: 54px;
    overflow: hidden;
    transition: border-color 150ms ease, box-shadow 150ms ease;
  }

  div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
    border-color: var(--eureka-amber);
    box-shadow: 0 0 0 3px rgba(213, 155, 69, 0.16);
  }

  div[data-testid="stTextInput"] input {
    background: transparent;
    border: 0;
    box-shadow: none;
    caret-color: var(--eureka-amber);
    color: var(--eureka-cream);
    font-size: 16px;
    padding: 13px 16px;
  }

  div[data-testid="stTextInput"] input::placeholder {
    color: rgba(216, 195, 154, 0.5);
    opacity: 1;
  }

  div[data-testid="stFormSubmitButton"] {
    margin-top: 8px;
  }

  div[data-testid="stFormSubmitButton"] button {
    background-color: var(--eureka-mustard);
    background-image: linear-gradient(180deg, #D59B45 0%, #B49A4B 100%);
    border: 1px solid rgba(241, 231, 211, 0.2);
    border-radius: 4px;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
    color: #1B1A15;
    font-family: "Oswald", "Arial Narrow", "Helvetica Neue", sans-serif !important;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    min-height: 52px;
    transition: background-color 150ms ease, border-color 150ms ease,
      box-shadow 150ms ease, filter 150ms ease;
  }

  div[data-testid="stFormSubmitButton"] button:hover {
    background-color: var(--eureka-amber);
    background-image: linear-gradient(180deg, #E0A64D 0%, #C3A64F 100%);
    border-color: rgba(241, 231, 211, 0.34);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.28);
    color: #14140F;
    filter: none;
  }

  div[data-testid="stFormSubmitButton"] button:active {
    background-color: var(--eureka-mustard);
    background-image: none;
    color: #14140F;
  }

  div[data-testid="stFormSubmitButton"] button:focus-visible {
    box-shadow: 0 0 0 3px rgba(213, 155, 69, 0.35);
    outline: none;
  }

  div[data-testid="stFormSubmitButton"] button p {
    color: inherit;
    font-family: inherit !important;
    font-size: inherit;
    font-weight: inherit;
    letter-spacing: inherit;
  }

  .eureka-status {
    align-items: center;
    color: rgba(216, 195, 154, 0.82);
    display: flex;
    font-size: 0.78rem;
    gap: 12px;
    justify-content: center;
    letter-spacing: 0.16em;
    line-height: 1.4;
    margin: 28px 0 0;
    text-transform: uppercase;
  }

  .eureka-spinner {
    animation: eureka-spin 0.9s linear infinite;
    border: 1px solid rgba(216, 195, 154, 0.2);
    border-radius: 999px;
    border-top-color: var(--eureka-amber);
    display: inline-block;
    flex: 0 0 auto;
    height: 15px;
    width: 15px;
  }

  @keyframes eureka-spin {
    to {
      transform: rotate(360deg);
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .eureka-spinner {
      animation-duration: 2.4s;
    }
  }

  .eureka-results {
    margin-top: 56px;
  }

  .eureka-results h2 {
    color: var(--eureka-sand);
    font-family: "Oswald", "Arial Narrow", "Helvetica Neue", sans-serif !important;
    font-size: 1.6rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    line-height: 1.2;
    margin: 0;
  }

  .eureka-results h2::after {
    background: var(--eureka-amber);
    content: "";
    display: block;
    height: 2px;
    margin-top: 12px;
    opacity: 0.85;
    width: 36px;
  }

  .eureka-list {
    border-top: 1px solid rgba(216, 195, 154, 0.16);
    list-style: none;
    margin: 20px 0 0;
    padding: 0;
  }

  .eureka-item {
    border-bottom: 1px solid rgba(216, 195, 154, 0.16);
    padding: 20px 0;
  }

  .stApp .eureka-name {
    color: var(--eureka-cream);
    font-size: 1.0625rem;
    font-weight: 500;
    line-height: 1.45;
    margin: 0 0 8px;
  }

  .eureka-num {
    color: var(--eureka-amber);
    font-family: "Oswald", "Arial Narrow", "Helvetica Neue", sans-serif !important;
    font-size: 0.95rem;
    font-variant-numeric: tabular-nums;
    font-weight: 500;
    letter-spacing: 0.06em;
    margin-right: 10px;
  }

  .eureka-watch {
    border-bottom: 1px solid rgba(108, 153, 145, 0.5);
    color: var(--eureka-turquoise);
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    padding-bottom: 1px;
    text-decoration: none;
    transition: border-color 150ms ease, color 150ms ease;
  }

  .eureka-watch:hover {
    border-color: var(--eureka-amber);
    color: var(--eureka-amber);
  }

  .eureka-watch:focus-visible {
    border-radius: 2px;
    outline: 2px solid var(--eureka-amber);
    outline-offset: 3px;
  }

  .stApp .eureka-empty {
    color: rgba(216, 195, 154, 0.48);
    font-size: 0.88rem;
    line-height: 1.5;
    margin: 0;
    padding: 4px 0 0;
  }

  .eureka-alert,
  .eureka-note {
    border-radius: 4px;
    font-size: 0.92rem;
    line-height: 1.55;
    margin-top: 24px;
    padding: 14px 18px;
    -webkit-backdrop-filter: blur(8px);
    backdrop-filter: blur(8px);
  }

  .eureka-alert {
    background: rgba(80, 32, 24, 0.72);
    border: 1px solid rgba(213, 110, 80, 0.4);
    color: #F2C9B8;
  }

  .eureka-note {
    background: rgba(52, 56, 44, 0.72);
    border: 1px solid rgba(216, 195, 154, 0.2);
    color: rgba(216, 195, 154, 0.86);
  }

  @media (max-width: 640px) {
    .stApp {
      background-attachment: scroll;
    }

    .block-container {
      padding: 7vh 20px 64px;
    }

    .eureka-brand {
      margin-bottom: 40px;
    }

    .eureka-hero {
      margin-bottom: 28px;
    }

    div[data-testid="stForm"] {
      padding: 16px;
    }

    .eureka-results {
      margin-top: 44px;
    }
  }
</style>
"""


@st.cache_resource
def _processing_slot() -> threading.BoundedSemaphore:
    return threading.BoundedSemaphore(1)


def _notice(message: str, tone: str = "error") -> None:
    class_name = "eureka-note" if tone == "note" else "eureka-alert"
    role = "status" if tone == "note" else "alert"
    st.markdown(
        f'<div class="{class_name}" role="{role}">{html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


def _show_status(placeholder: Any, message: str) -> None:
    placeholder.markdown(
        '<div class="eureka-status" role="status">'
        '<span class="eureka-spinner" aria-hidden="true"></span>'
        f"<span>{html.escape(message)}</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _friendly_stage(stage: str) -> str:
    text = (stage or "").casefold()
    if "saving recommendation" in text:
        return FINISHING_MESSAGE
    if "youtube" in text:
        return YOUTUBE_MESSAGE
    if "recommend" in text:
        return UNDERSTANDING_MESSAGE
    return PROCESSING_MESSAGE


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, ReelValidationError):
        return str(exc) or "That doesn't look like a valid Instagram Reel link."

    if isinstance(exc, ReelDownloadError):
        messages = {
            "unavailable_or_login": (
                "This Reel is unavailable, private, or requires login. Instagram may "
                "also be blocking downloads from this server. Only public Reels can "
                "be processed."
            ),
            "age_restricted": "This Reel is age-restricted and can't be processed.",
            "rate_limited": (
                "Instagram is rate-limiting requests from this server right now. "
                "Please try again in a moment."
            ),
            "login_required": (
                "This Reel requires an Instagram login, so it can't be downloaded "
                "from this server."
            ),
            "private": "This Reel is from a private account.",
            "unsupported_url": "That link isn't a supported Instagram Reel.",
            "no_format": "We couldn't find downloadable media in this Reel.",
            "too_large": "This Reel is too long to process. Try a shorter Reel.",
            "too_long": "This Reel is too long to process. Try a shorter Reel.",
            "ffmpeg_missing": (
                "The server's media tools are unavailable. Please try again later."
            ),
            "internal": (
                "Eureka isn't available right now. Please try again later."
            ),
            "invalid_media": "We couldn't prepare this Reel. Please try another Reel.",
            "access_denied": (
                "Instagram denied the download request. This can happen when "
                "Instagram blocks cloud server IPs. Please try again later."
            ),
            "unavailable": "This Reel is unavailable or has been removed.",
            "timeout": "This Reel took too long to load. Please try again.",
        }
        return messages.get(
            exc.category,
            "We couldn't download this Reel. It may be private, unavailable, or "
            "blocked for this server.",
        )

    if isinstance(exc, TranscriptionError):
        if exc.category in {"missing_dependency", "model_unavailable"}:
            return "Eureka isn't available right now. Please try again later."
        if exc.category in {"audio_missing", "invalid_audio", "empty_audio"}:
            return "We couldn't read the audio in this Reel. Please try another Reel."
        return "We couldn't transcribe this Reel. Please try again."

    if isinstance(exc, StorageError):
        if exc.category == "busy":
            return "This Reel is already being processed. Try again in a moment."
        return "We couldn't save this Reel's results. Please try again."

    return "Something went wrong while processing this Reel. Please try again."


def _recommendation_title(match: RecommendationMatch) -> str:
    recommendation = match.recommendation
    video = match.videos[0] if match.videos else None

    title = (recommendation.title or "").strip()
    if not title and video is not None:
        title = (video.title or "").strip()
    return title or "Recommended video"


def _render_recommendation(index: int, match: RecommendationMatch) -> str:
    title = _recommendation_title(match)
    name = f'<span class="eureka-num">{index}.</span>{html.escape(title)}'

    action = ""
    if match.videos:
        url = html.escape(match.videos[0].url, quote=True)
        action = (
            f'<a class="eureka-watch" href="{url}" target="_blank" '
            'rel="noopener noreferrer">Watch on YouTube '
            '<span aria-hidden="true">↗</span></a>'
        )

    return f'<li class="eureka-item"><p class="eureka-name">{name}</p>{action}</li>'


def _render_results(result: ProcessedReel) -> None:
    if not result.has_audio:
        _notice("This Reel has no audio, so there are no recommendations to find.", "note")
        return

    if not result.transcript:
        _notice("No speech was found in this Reel.", "note")
        return

    if result.is_recommendation_reel is False:
        _notice(
            "This Reel doesn't recommend any specific content, "
            "so there are no videos to search for."
        )
        return

    if result.recommendation_error and not result.recommendations:
        _notice(
            "We couldn't find recommendations in this Reel right now. "
            "Please try again.",
        )
        return

    if not result.recommendations:
        st.markdown(
            '<section class="eureka-results"><h2>Your discoveries</h2>'
            '<p class="eureka-empty">No discoveries found in this Reel.</p>'
            "</section>",
            unsafe_allow_html=True,
        )
        return

    items = "".join(
        _render_recommendation(index, match)
        for index, match in enumerate(result.recommendations, start=1)
    )
    st.markdown(
        '<section class="eureka-results"><h2>Your discoveries</h2>'
        f'<ol class="eureka-list">{items}</ol></section>',
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Eureka — Discover what you hear.",
    page_icon=str(FAVICON_PATH),
    layout="centered",
)

st.markdown(STYLES, unsafe_allow_html=True)

st.markdown(
    '<meta name="description" content="Eureka — Discover what you hear. '
    "Paste an Instagram Reel. Find the recommendations hidden inside it.\">"
    '<meta name="application-name" content="Eureka">'
    '<meta name="color-scheme" content="dark">'
    '<meta name="theme-color" content="#191B18">'
    '<meta property="og:type" content="website">'
    '<meta property="og:site_name" content="Eureka">'
    '<meta property="og:title" content="Eureka — Discover what you hear.">'
    '<meta property="og:description" content="Paste an Instagram Reel. '
    'Find the recommendations hidden inside it.">'
    '<meta name="twitter:card" content="summary">'
    '<meta name="twitter:title" content="Eureka — Discover what you hear.">'
    '<meta name="twitter:description" content="Paste an Instagram Reel. '
    'Find the recommendations hidden inside it.">',
    unsafe_allow_html=True,
)

st.markdown(
    '<header class="eureka-brand">'
    '<div class="eureka-brand-row">'
    '<svg class="eureka-mark" viewBox="0 0 40 26" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">'
    '<circle cx="20" cy="11" r="6.5" fill="#D59B45"/>'
    '<path d="M1.5 20.5h10.5l3.5-4.5 4.5 4.5H38.5" stroke="#D8C39A" '
    'stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M1.5 24.5h37" stroke="#6C9991" stroke-width="1" '
    'stroke-linecap="round" opacity="0.8"/>'
    "</svg>"
    '<span class="eureka-wordmark">Eureka</span>'
    "</div>"
    '<p class="eureka-tagline">Discover what you hear.</p>'
    "</header>",
    unsafe_allow_html=True,
)

st.markdown(
    '<section class="eureka-hero">'
    "<h1>Found something interesting?</h1>"
    '<p class="eureka-description">Paste an Instagram Reel. '
    "Find the recommendations hidden inside it.</p>"
    "</section>",
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None

with st.form("reel_recommendation_form"):
    reel_url = st.text_input(
        "Instagram Reel link",
        placeholder="Paste Instagram Reel link here",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button(
        "Find recommendations",
        use_container_width=True,
    )

st.markdown(
    '<p class="eureka-disclaimer">Video matching depends on audio quality. '
    "Eureka can find the exact video only when the creator reads the full name; "
    "otherwise, it shows the closest match.</p>",
    unsafe_allow_html=True,
)

if submitted:
    st.session_state.result = None
    st.session_state.error = None
    slot = _processing_slot()

    if not reel_url.strip():
        st.session_state.error = "Paste an Instagram Reel link to get started."
    elif not slot.acquire(blocking=False):
        st.session_state.error = "Eureka is busy right now. Try again in a moment."
    else:
        status_placeholder = st.empty()
        _show_status(status_placeholder, PROCESSING_MESSAGE)
        try:
            def on_stage(label: str) -> None:
                _show_status(status_placeholder, _friendly_stage(label))

            try:
                result = process_reel(reel_url.strip(), on_stage=on_stage)
            except ReelValidationError as exc:
                logger.info("Reel validation failed")
                st.session_state.error = _friendly_error(exc)
            except StorageError as exc:
                logger.warning("Local storage failure category=%s", exc.category)
                st.session_state.error = _friendly_error(exc)
            except TranscriptionError as exc:
                logger.warning("Transcription failure category=%s", exc.category)
                st.session_state.error = _friendly_error(exc)
            except ReelDownloadError as exc:
                logger.warning("Audio retrieval failure category=%s", exc.category)
                st.session_state.error = _friendly_error(exc)
            except Exception as exc:
                logger.error(
                    "Unexpected error while processing Reel: %s",
                    type(exc).__name__,
                )
                st.session_state.error = _friendly_error(exc)
            else:
                st.session_state.result = result
        finally:
            status_placeholder.empty()
            slot.release()

if st.session_state.error:
    _notice(st.session_state.error)

if st.session_state.result:
    _render_results(st.session_state.result)
