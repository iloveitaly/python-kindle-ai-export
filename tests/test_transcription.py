"""Tests for transcription orchestration, retries, error handling, and checkpoint caching."""

import json
from pathlib import Path
from unittest.mock import AsyncMock
import pytest
from python_kindle_ai_export.models import PageChunk
from python_kindle_ai_export.transcription import (
    INITIAL_BACKOFF_MS,
    MAX_BACKOFF_MS,
    MAX_RETRY_AFTER_MS,
    TranscriptionError,
    TranscriptionResult,
    calculate_retry_delay,
    parse_transcription,
    transcribe_pages,
    with_retries,
)


@pytest.fixture
def transcription_fixture(tmp_path: Path):
    out_dir = tmp_path / "transcribe_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshot = out_dir / "page.png"
    screenshot.write_bytes(b"image-one")
    pages = [PageChunk(index=0, page=1, screenshot=str(screenshot))]
    return out_dir, pages


def test_distinguishes_blank_from_missing_output_and_preserves_literal_text() -> None:
    blank_res = parse_transcription("<BLANK_PAGE>")
    assert blank_res.status == "blank"
    assert blank_res.text == ""

    with pytest.raises(TranscriptionError, match="no text"):
        parse_transcription("")

    with pytest.raises(TranscriptionError, match="no text"):
        parse_transcription("   ")

    success_res = parse_transcription("123\nA heading")
    assert success_res.status == "success"
    assert success_res.text == "123\nA heading"


@pytest.mark.asyncio
async def test_bounds_transient_retries_and_honors_retry_after() -> None:
    calls = 0

    async def op() -> TranscriptionResult:
        nonlocal calls
        calls += 1
        raise TranscriptionError("rate limit", kind="failed", retryable=True, retry_after_ms=9000)

    sleep_mock = AsyncMock()
    with pytest.raises(TranscriptionError, match="rate limit"):
        await with_retries(op, max_attempts=3, timeout_ms=100, sleep_fn=sleep_mock)

    assert calls == 3
    # Verify that sleep was called with 9000ms / 1000 = 9.0s
    sleep_mock.assert_called_with(9.0)


@pytest.mark.asyncio
async def test_does_not_retry_blocked_output() -> None:
    calls = 0

    async def op() -> TranscriptionResult:
        nonlocal calls
        calls += 1
        raise TranscriptionError("blocked", kind="blocked", retryable=False)

    with pytest.raises(TranscriptionError, match="blocked"):
        await with_retries(op, max_attempts=3, timeout_ms=100)

    assert calls == 1


@pytest.mark.asyncio
async def test_aborts_requests_that_exceed_timeout() -> None:
    async def op() -> TranscriptionResult:
        import asyncio
        await asyncio.sleep(0.5)
        return TranscriptionResult(status="success", text="ok")

    with pytest.raises(TranscriptionError, match="timed out"):
        await with_retries(op, max_attempts=1, timeout_ms=5)


@pytest.mark.asyncio
async def test_stops_new_requests_after_permanent_error(transcription_fixture) -> None:
    out_dir, pages = transcription_fixture
    second = out_dir / "second.png"
    second.write_bytes(b"image-two")
    pages.append(PageChunk(index=1, page=2, screenshot=str(second)))

    mock_provider = AsyncMock()
    mock_provider.name = "mock"
    mock_provider.model = "one"
    mock_provider.transcribe.side_effect = TranscriptionError(
        "invalid credentials",
        kind="failed",
        retryable=False,
        fatal=True,
    )

    with pytest.raises(ValueError, match="incomplete"):
        await transcribe_pages(
            pages=pages,
            out_dir=out_dir,
            provider=mock_provider,
            concurrency=1,
            max_attempts=2,
            timeout_ms=100,
        )

    assert mock_provider.transcribe.call_count == 1


@pytest.mark.asyncio
async def test_reuses_checkpointed_pages_and_invalidates_on_change(transcription_fixture) -> None:
    out_dir, pages = transcription_fixture
    mock_provider = AsyncMock()
    mock_provider.name = "mock"
    mock_provider.model = "one"
    mock_provider.transcribe.return_value = TranscriptionResult(status="success", text="Book text")

    # First run
    await transcribe_pages(pages=pages, out_dir=out_dir, provider=mock_provider, concurrency=1, max_attempts=2, timeout_ms=100)
    assert mock_provider.transcribe.call_count == 1

    # Second run with same data uses cache
    await transcribe_pages(pages=pages, out_dir=out_dir, provider=mock_provider, concurrency=1, max_attempts=2, timeout_ms=100)
    assert mock_provider.transcribe.call_count == 1

    # Modified image invalidates cache
    Path(pages[0].screenshot).write_bytes(b"image-two")
    await transcribe_pages(pages=pages, out_dir=out_dir, provider=mock_provider, concurrency=1, max_attempts=2, timeout_ms=100)
    assert mock_provider.transcribe.call_count == 2


@pytest.mark.asyncio
async def test_removes_stale_output_on_failure_and_resumes(transcription_fixture) -> None:
    out_dir, pages = transcription_fixture
    second = out_dir / "second.png"
    second.write_bytes(b"second")
    pages.append(PageChunk(index=1, page=2, screenshot=str(second)))

    # Write stale content
    (out_dir / "content.json").write_text('["stale"]', encoding="utf-8")

    mock_provider = AsyncMock()
    mock_provider.name = "mock"
    mock_provider.model = "one"
    mock_provider.transcribe.side_effect = [
        TranscriptionResult(status="success", text="First"),
        TranscriptionError("blocked", kind="blocked", retryable=False),
    ]

    with pytest.raises(ValueError, match="incomplete"):
        await transcribe_pages(pages=pages, out_dir=out_dir, provider=mock_provider, concurrency=1, max_attempts=1, timeout_ms=100)

    # Content.json should have been unlinked
    assert not (out_dir / "content.json").exists()

    report_raw = json.loads((out_dir / "transcription.json").read_text(encoding="utf-8"))
    assert report_raw["status"] == "partial"
    assert report_raw["pages"][1]["status"] == "blocked"

    # Resume run with fixed second call
    mock_provider.transcribe.side_effect = [
        TranscriptionResult(status="blank", text=""),
    ]
    content = await transcribe_pages(pages=pages, out_dir=out_dir, provider=mock_provider, concurrency=1, max_attempts=1, timeout_ms=100)
    assert [c.text for c in content] == ["First", ""]


def test_calculate_retry_delay_exponential_and_caps() -> None:
    # Attempt 1: 2s base -> [1.0s, 2.0s]
    assert calculate_retry_delay(1, randomizer=lambda: 0.0) == 1.0
    assert calculate_retry_delay(1, randomizer=lambda: 1.0) == 2.0

    # Attempt 2: 4s base -> [2.0s, 4.0s]
    assert calculate_retry_delay(2, randomizer=lambda: 0.0) == 2.0
    assert calculate_retry_delay(2, randomizer=lambda: 1.0) == 4.0

    # Attempt 10: 2000 * 2^9 = 1,024,000ms capped at MAX_BACKOFF_MS (900_000ms = 15m)
    assert calculate_retry_delay(10, randomizer=lambda: 1.0) == 900.0
    assert calculate_retry_delay(10, randomizer=lambda: 0.0) == 450.0

    # Server Retry-After overrides exponential backoff if larger
    assert calculate_retry_delay(1, retry_after_ms=60_000, randomizer=lambda: 0.0) == 60.0

    # Server Retry-After capped at MAX_RETRY_AFTER_MS (3600s = 1h)
    assert calculate_retry_delay(1, retry_after_ms=7_200_000, randomizer=lambda: 0.0) == 3600.0


@pytest.mark.asyncio
async def test_with_retries_recovers_from_generic_exception() -> None:
    calls = 0

    async def op() -> TranscriptionResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionResetError("Connection lost")
        return TranscriptionResult(status="success", text="Recovered")

    sleep_mock = AsyncMock()
    result = await with_retries(
        op,
        max_attempts=3,
        timeout_ms=1000,
        sleep_fn=sleep_mock,
        random_fn=lambda: 1.0,
    )

    assert result.text == "Recovered"
    assert calls == 2
    sleep_mock.assert_called_once_with(2.0)


@pytest.mark.asyncio
async def test_with_retries_recovers_from_timeout() -> None:
    calls = 0

    async def op() -> TranscriptionResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            import asyncio
            await asyncio.sleep(0.1)
        return TranscriptionResult(status="success", text="Timeout recovered")

    sleep_mock = AsyncMock()
    result = await with_retries(
        op,
        max_attempts=3,
        timeout_ms=50,
        sleep_fn=sleep_mock,
        random_fn=lambda: 1.0,
    )

    assert result.text == "Timeout recovered"
    assert calls == 2
    sleep_mock.assert_called_once_with(2.0)

