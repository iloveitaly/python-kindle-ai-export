[![Release Notes](https://img.shields.io/github/release/iloveitaly/python-kindle-ai-export)](https://github.com/iloveitaly/python-kindle-ai-export/releases)
[![Downloads](https://static.pepy.tech/badge/python-kindle-ai-export/month)](https://pepy.tech/project/python-kindle-ai-export)
![GitHub CI Status](https://github.com/iloveitaly/python-kindle-ai-export/actions/workflows/build_and_publish.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Export Kindle Books to Markdown, PDF, and Audiobooks

Automate the extraction of your Kindle library from Amazon Cloud Reader into clean Markdown files, searchable PDFs, and audiobooks. Uses Camoufox anti-detect browser automation to capture rendered book pages, Pydantic-AI to transcribe screens with vision models, and TTS engines to generate narrations.

## Installation

```bash
uv add python-kindle-ai-export
```

## Usage

### 1. Extract Book Pages

Capture rendered screens and book metadata from the Kindle Cloud Reader:

```bash
uv run python-kindle-ai-export extract --asin B00EXAMPLE --email "you@example.com" --password "secret"
```

#### Manual login (recommended when 2FA is enabled)

Omit the email/password (or pass `--manual-login`) and sign in yourself, including 2FA, in the browser window. The extractor waits until the reader loads, then continues automatically. The session is saved in the browser profile under the output directory, so later runs with the same `--out-dir` skip login. `EXTRACT_LOGIN_TIMEOUT` (default `600` seconds) caps the wait.

```bash
uv run python-kindle-ai-export extract --asin B00EXAMPLE
```

#### Human-like pacing

Extraction paces browser actions to look like a person reading: each click, key press, or navigation gets a random delay, and after a capped number of actions in a rolling window the extractor takes a multi-minute "reading break". This makes extraction much slower (roughly 35–40 minutes for a 300-screen book). Ctrl+C cuts a break short.

| Env var | Default | Description |
|---|---|---|
| `EXTRACT_PACING` | `on` | Set to `off` to disable all pacing |
| `EXTRACT_ACTION_DELAY` | `0.8-3.0` | Random delay before each action, in seconds |
| `EXTRACT_WINDOW_SECONDS` | `300` | Length of the rolling action window |
| `EXTRACT_MAX_ACTIONS_PER_WINDOW` | `50-70` | Action cap per window (re-rolled after each break) |
| `EXTRACT_BREAK_SECONDS` | `120-240` | Break length once the cap is reached |

Like the other settings, these also accept the `KINDLE_AI_EXPORT_` / `KINDLE_EXPORT_` prefixes.

#### Resuming interrupted extractions

Extraction resumes automatically by default if an existing `metadata.json` checkpoint is found in the output directory. It skips the initial rewind loop, jumps directly to the last captured position, advances to the next uncaptured page, and appends remaining screens. If an extraction is already marked `complete`, it returns immediately without re-navigating.

Pass `--no-resume` (or set `EXTRACT_RESUME=false` / `KINDLE_AI_EXPORT_RESUME=false`) to force a fresh restart from the beginning.

### 2. Transcribe Pages with AI

Transcribe captured page screenshots into clean text using vision models via Pydantic-AI (defaults to `gemini-3.8-flash`):

```bash
# Using environment variables
export KINDLE_AI_EXPORT_AI_KEY="AIzaSy..."
uv run python-kindle-ai-export transcribe --asin B00EXAMPLE

# Or specifying a custom model
uv run python-kindle-ai-export transcribe --asin B00EXAMPLE --model openai:gpt-4o
```

### 3. Export Formats

Export your transcribed content to Markdown, PDF, or audio:

```bash
# Markdown
uv run python-kindle-ai-export export-markdown --asin B00EXAMPLE

# PDF (auto, text, or images)
uv run python-kindle-ai-export export-pdf --asin B00EXAMPLE --mode auto

# Audiobook (OpenAI or UnrealSpeech TTS with ID3 tags)
uv run python-kindle-ai-export export-audio --asin B00EXAMPLE --engine openai

# All document formats
uv run python-kindle-ai-export export-all --asin B00EXAMPLE
```

## Features

- Anti-detect browser capture powered by Camoufox with automatic session restoration and coordinate tracking.
- Multi-provider AI vision transcription powered by Pydantic-AI (OpenAI, Google Gemini, Anthropic).
- Clean Markdown export with table of contents anchoring and heading hierarchies.
- Searchable text and image PDF generation via ReportLab.
- Audiobook synthesis via OpenAI TTS or UnrealSpeech with automatic audio chunking and ID3 metadata tagging.
- Resumable workflows with atomic JSON checkpoints and cache digests.

## [MIT License](LICENSE.md)

---

*This project was created from [iloveitaly/python-package-template](https://github.com/iloveitaly/python-package-template)*