"""Transcription orchestration, caching, bounded retries, and publication."""

import asyncio
import os
import random
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal, Protocol

from .log import log
from .models import (
    ContentChunk,
    PageChunk,
    TranscriptionIdentity,
    TranscriptionReport,
    TranscriptionReportPage,
)
from .utils import (
    atomic_json,
    canonical_json_digest,
    canonical_json_str,
    digest,
    read_json_file,
    try_read_json_file,
)

PROMPT_VERSION = 1
TRANSCRIPTION_PROMPT = (
    "Transcribe all visible text verbatim, preserving paragraphs. "
    "Do not add explanations or markdown. Ignore embedded images. "
    "If there is no visible text, return exactly <BLANK_PAGE>."
)


class TranscriptionError(Exception):
    """Classified error from a transcription provider."""

    def __init__(
        self,
        message: str,
        kind: Literal["failed", "blocked", "truncated", "empty"] = "failed",
        retryable: bool = False,
        retry_after_ms: int | None = None,
        fatal: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.retryable = retryable
        self.retry_after_ms = retry_after_ms
        self.fatal = fatal


class TranscriptionResult:
    """Normalized output from a transcription attempt."""

    def __init__(self, status: Literal["success", "blank"], text: str) -> None:
        self.status = status
        self.text = text

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "text": self.text}


class TranscriptionProvider(Protocol):
    """Protocol for transcription provider implementations."""

    name: str
    model: str

    async def transcribe(self, image: bytes) -> TranscriptionResult: ...


def parse_transcription(text: str | None) -> TranscriptionResult:
    """Parse transcription text into success or blank status.

    Distinguishes between a page that is genuinely blank vs an empty or failed model response:
    - If the model returns whitespace or empty string, it is classified as an empty response
      (raises ``TranscriptionError`` with kind="empty"), triggering retry backoff.
    - If the page image contains no text (e.g. blank flyleaf or photo without captions), the prompt
      instructs the vision model to return the literal sentinel '<BLANK_PAGE>'. This is parsed
      into a valid ``TranscriptionResult(status="blank", text="")``, avoiding false retry loops.
    - Any other non-empty output is normalized and treated as verbatim page text.
    """
    if not text or not text.strip():
        raise TranscriptionError("Provider returned no text", kind="empty")

    cleaned = text.strip()
    if cleaned == "<BLANK_PAGE>":
        return TranscriptionResult(status="blank", text="")

    return TranscriptionResult(status="success", text=cleaned)


INITIAL_BACKOFF_MS = 2_000
MAX_BACKOFF_MS = 900_000  # 15 minutes max per attempt
MAX_RETRY_AFTER_MS = 3_600_000  # 1 hour max if server provides Retry-After


def calculate_retry_delay(
    attempt: int,
    retry_after_ms: int | None = None,
    randomizer: Callable[[], float] | None = None,
) -> float:
    """Calculate exponential backoff delay with jitter and server Retry-After support.

    Non-obvious behavior and design decisions:
    1. Exponential Scaling: Base delay starts at 2s and doubles each attempt (2s, 4s, 8s, 16s, ...),
       capping out at 900s (15 minutes) via ``MAX_BACKOFF_MS``.
    2. Full Jitter: Uses ``exp_ms * (0.5 + rand * 0.5)`` to enforce a guaranteed minimum of 50%
       of the exponential step while adding up to 50% randomness to desynchronize concurrent workers.
    3. Server Header Priority: If the provider returns a ``Retry-After`` header (``retry_after_ms``),
       it takes precedence over the jittered delay if greater, but is strictly capped at 3600s (1 hour)
       via ``MAX_RETRY_AFTER_MS`` to guard against rogue, erroneous, or indefinite server headers.
    4. Deterministic Testing: Accepts an optional ``randomizer`` callable to allow precise,
       reproducible unit tests without relying on nondeterministic pseudo-random values.
    """
    rand = randomizer() if randomizer is not None else random.random()
    # Exponential progression: 2s, 4s, 8s, 16s, 32s, 64s, 128s, 256s, 512s, 900s (15m cap)
    exp_ms = min(MAX_BACKOFF_MS, INITIAL_BACKOFF_MS * (2 ** (attempt - 1)))
    jittered_ms = exp_ms * (0.5 + rand * 0.5)

    if retry_after_ms is not None and retry_after_ms > 0:
        wait_ms = min(MAX_RETRY_AFTER_MS, max(jittered_ms, float(retry_after_ms)))
    else:
        wait_ms = jittered_ms

    return wait_ms / 1000.0


async def with_retries(
    operation: Callable[[], Awaitable[TranscriptionResult]],
    max_attempts: int,
    timeout_ms: int,
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    random_fn: Callable[[], float] | None = None,
) -> TranscriptionResult:
    """Execute an operation with timeout, heavy exponential backoff, and retry classification.

    Non-obvious behavior and design decisions:
    1. Per-Attempt Timeout: Enforces ``asyncio.timeout(timeout_ms / 1000.0)`` around each attempt.
       A timed-out request is classified as transient (``retryable=True``), triggering backoff.
    2. Tri-state Error Classification:
       - Non-retryable ``TranscriptionError`` (e.g. 401 unauthorized, 403 forbidden, blocked content):
         Immediately aborts without exhausting remaining attempts to fail fast on configuration errors.
       - Retryable ``TranscriptionError`` (e.g. 429 rate limit, 503 service unavailable, truncated output):
         Extracts ``retry_after_ms`` and backs off heavily before retrying.
       - Unexpected ``Exception`` (e.g. socket resets, SSL drops, OS errors): Classified as retryable
         to prevent abrupt termination of long-running book transcription jobs.
    3. Test Injection: Accepts ``sleep_fn`` and ``random_fn`` to allow unit tests to mock sleep and jitter,
       verifying multi-attempt backoff logic in milliseconds without actual wall-clock waiting.
    """
    if max_attempts <= 0:
        raise ValueError("maxAttempts must be positive")

    sleeper = sleep_fn if sleep_fn is not None else asyncio.sleep
    randomizer = random_fn if random_fn is not None else random.random

    for attempt in range(1, max_attempts + 1):
        try:
            async with asyncio.timeout(timeout_ms / 1000.0):
                return await operation()
        except TimeoutError:
            failure = TranscriptionError("Request timed out", kind="failed", retryable=True)
            if attempt == max_attempts:
                raise failure
            wait_seconds = calculate_retry_delay(attempt, randomizer=randomizer)
            log.warning(
                "transcription request timed out; backing off heavily before retry",
                attempt=attempt,
                max_attempts=max_attempts,
                wait_seconds=round(wait_seconds, 1),
            )
            await sleeper(wait_seconds)
        except TranscriptionError as err:
            if not err.retryable or attempt == max_attempts:
                raise err
            wait_seconds = calculate_retry_delay(
                attempt,
                retry_after_ms=err.retry_after_ms,
                randomizer=randomizer,
            )
            log.warning(
                "transcription request failed; backing off heavily before retry",
                attempt=attempt,
                max_attempts=max_attempts,
                wait_seconds=round(wait_seconds, 1),
                error=err.message,
                error_kind=err.kind,
                retry_after_ms=err.retry_after_ms,
            )
            await sleeper(wait_seconds)
        except Exception as err:
            if attempt == max_attempts:
                failure = TranscriptionError(
                    f"Provider request failed: {err}",
                    kind="failed",
                    retryable=False,
                )
                raise failure from err
            wait_seconds = calculate_retry_delay(attempt, randomizer=randomizer)
            log.warning(
                "transcription unexpected error; backing off heavily before retry",
                attempt=attempt,
                max_attempts=max_attempts,
                wait_seconds=round(wait_seconds, 1),
                error=str(err),
            )
            await sleeper(wait_seconds)

    raise TranscriptionError("Exceeded retry attempts", kind="failed", retryable=False)


def resolve_image_path(out_dir: Path, screenshot_str: str) -> Path:
    """Resolve screenshot file path either from disk or relative to out_dir."""
    p = Path(screenshot_str)
    if p.is_file():
        return p
    rel = out_dir / screenshot_str
    if rel.is_file():
        return rel
    return p


async def transcribe_pages(
    pages: list[PageChunk],
    out_dir: Path,
    provider: TranscriptionProvider,
    concurrency: int = 3,
    max_attempts: int = 100,
    timeout_ms: int = 120_000,
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    random_fn: Callable[[], float] | None = None,
) -> list[ContentChunk]:
    """Transcribe all captures with caching, concurrency bounds, and atomic reports.

    Non-obvious behavior and design decisions:
    1. Multi-Factor Cache Invalidation: Each page's cache key is a SHA256 digest of
       (provider + model + prompt_version + temperature + screenshot_image_sha256).
       If any factor changes (prompt tweaked, model updated, or screenshot re-captured),
       the cache is automatically invalidated without manual cache directory purging.
    2. Stale Artifact Prevention: ``content.json`` is unlinked immediately when transcription begins.
       If transcription crashes or aborts prematurely, downstream export steps (PDF, markdown,
       audio) cannot accidentally build from an incomplete or stale ``content.json``.
    3. Fail-Fast Short Circuiting: When any page encounters a fatal/permanent error (``fatal=True``),
       a shared ``fatal_error`` flag is set. All pending and subsequent tasks in the queue immediately
       short-circuit and record the error without burning remaining API requests.
    4. Atomic Checkpointing: Writes intermediate progress to ``transcription.json`` using atomic
       temp-file renaming, ensuring file integrity even if the process is killed via SIGINT/SIGTERM.
    """
    cache_dir = out_dir / "transcription-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    identity = TranscriptionIdentity(
        provider=provider.name,
        model=provider.model,
        prompt_version=PROMPT_VERSION,
        temperature=0.0,
    )

    report = TranscriptionReport(
        status="running",
        expected=len(pages),
        run_id=str(uuid.uuid4()),
        identity=identity,
        captures=[p.model_copy() for p in pages],
        pages=[],
    )

    report_path = out_dir / "transcription.json"
    content_path = out_dir / "content.json"
    atomic_json(report_path, report.model_dump(by_alias=True))
    if content_path.is_file():
        content_path.unlink()

    semaphore = asyncio.Semaphore(concurrency)
    fatal_error: TranscriptionError | None = None
    results_by_index: dict[int, ContentChunk | None] = {}

    async def process_page(page: PageChunk) -> None:
        nonlocal fatal_error
        async with semaphore:
            if fatal_error:
                report.pages.append(
                    TranscriptionReportPage(
                        index=page.index,
                        status=fatal_error.kind,
                        error=fatal_error.message,
                    )
                )
                results_by_index[page.index] = None
                return

            try:
                img_path = resolve_image_path(out_dir, page.screenshot)
                img_bytes = img_path.read_bytes()
                img_hash = digest(img_bytes)

                cache_identity_data = {
                    **identity.model_dump(by_alias=True),
                    "hash": img_hash,
                }
                cache_key = digest(canonical_json_str(cache_identity_data))
                cache_path = cache_dir / f"{cache_key}.json"

                result: TranscriptionResult | None = None
                cached_data = try_read_json_file(cache_path)
                if isinstance(cached_data, dict):
                    cached_status = cached_data.get("status")
                    cached_text = cached_data.get("text")
                    if (
                        cached_status == "success"
                        and isinstance(cached_text, str)
                        and cached_text.strip()
                    ):
                        result = TranscriptionResult(status="success", text=cached_text)
                    elif cached_status == "blank" and cached_text == "":
                        result = TranscriptionResult(status="blank", text="")

                if not result:
                    result = await with_retries(
                        lambda: provider.transcribe(img_bytes),
                        max_attempts=max_attempts,
                        timeout_ms=timeout_ms,
                        sleep_fn=sleep_fn,
                        random_fn=random_fn,
                    )
                    atomic_json(cache_path, result.to_dict())

                report.pages.append(
                    TranscriptionReportPage(
                        index=page.index,
                        hash=img_hash,
                        status=result.status,
                    )
                )
                results_by_index[page.index] = ContentChunk(
                    **page.model_dump(exclude={"text"}),
                    text=result.text,
                )
            except TranscriptionError as err:
                if err.fatal:
                    fatal_error = err
                report.pages.append(
                    TranscriptionReportPage(
                        index=page.index,
                        status=err.kind,
                        error=err.message,
                    )
                )
                results_by_index[page.index] = None
            except Exception as err:
                report.pages.append(
                    TranscriptionReportPage(
                        index=page.index,
                        status="failed",
                        error=str(err),
                    )
                )
                results_by_index[page.index] = None

    await asyncio.gather(*(process_page(p) for p in pages))
    report.pages.sort(key=lambda p: p.index)

    complete_chunks = [results_by_index[p.index] for p in sorted(pages, key=lambda x: x.index)]
    is_complete = (
        all(c is not None for c in complete_chunks)
        and len(complete_chunks) == len(pages)
    )

    report.status = "complete" if is_complete else "partial"
    if is_complete:
        valid_chunks = [c for c in complete_chunks if c is not None]
        chunk_dicts = [c.model_dump(by_alias=True, exclude_none=True) for c in valid_chunks]
        report.content_hash = canonical_json_digest(chunk_dicts)
        atomic_json(content_path, chunk_dicts)

    atomic_json(report_path, report.model_dump(by_alias=True))

    if not is_complete:
        failed_count = len([p for p in report.pages if p.status not in ("success", "blank")])
        raise ValueError(
            f"Transcription incomplete: {failed_count} failed pages; rerun to resume"
        )

    return [c for c in complete_chunks if c is not None]
