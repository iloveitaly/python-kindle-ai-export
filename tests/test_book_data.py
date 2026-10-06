"""Tests for book data loading, completeness assertions, and legacy compatibility."""

import json
from pathlib import Path
import pytest
from python_kindle_ai_export.book_data import (
    assert_complete_extraction,
    assert_complete_transcription,
    read_book_content,
    read_book_metadata,
)
from python_kindle_ai_export.models import (
    AmazonBookMeta,
    BookMetadata,
    ContentChunk,
    ExtractionStatus,
    PageChunk,
    TranscriptionIdentity,
    TranscriptionReport,
    TranscriptionReportPage,
)
from python_kindle_ai_export.utils import canonical_json_digest, digest, try_read_json_file


@pytest.fixture
def book_fixture(tmp_path: Path):
    out_dir = tmp_path / "book_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshot = out_dir / "page.png"
    screenshot.write_bytes(b"page-image")

    page = PageChunk(index=0, page=1, screenshot=str(screenshot))
    metadata = BookMetadata(
        meta=AmazonBookMeta(asin="BOOK", title="Book", author_list=[]),
        schema_version=None,
        toc=[],
        pages=[page],
    )
    content = [ContentChunk(**page.model_dump(), text="Last chapter")]
    return out_dir, metadata, content


def test_loads_legacy_data_without_sidecar(book_fixture) -> None:
    out_dir, metadata, content = book_fixture
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata.model_dump(by_alias=True)), encoding="utf-8"
    )
    (out_dir / "content.json").write_text(
        json.dumps([c.model_dump(by_alias=True) for c in content]), encoding="utf-8"
    )

    loaded = read_book_metadata(out_dir / "metadata.json")
    chunks = read_book_content(out_dir / "content.json")
    assert_complete_transcription(out_dir, loaded, chunks)


def test_rejects_duplicate_captures_and_unsupported_schema(book_fixture) -> None:
    out_dir, metadata, _ = book_fixture
    meta_path = out_dir / "metadata.json"

    meta_dict = metadata.model_dump(by_alias=True)
    meta_dict["schemaVersion"] = 999
    meta_path.write_text(json.dumps(meta_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        read_book_metadata(meta_path)

    dup_dict = metadata.model_dump(by_alias=True)
    dup_dict["pages"] = dup_dict["pages"] + dup_dict["pages"]
    meta_path.write_text(json.dumps(dup_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_book_metadata(meta_path)


def test_rejects_partial_extraction_and_missing_completion(book_fixture) -> None:
    _, metadata, _ = book_fixture
    metadata.extraction = ExtractionStatus(status="partial", reason="stalled")
    with pytest.raises(ValueError, match="stalled"):
        assert_complete_extraction(metadata)

    metadata.extraction = None
    metadata.schema_version = 2
    with pytest.raises(ValueError, match="completion"):
        assert_complete_extraction(metadata)


def test_checks_current_image_hashes_and_rejects_tampered_text(book_fixture) -> None:
    out_dir, metadata, content = book_fixture
    metadata.schema_version = 2
    metadata.extraction = ExtractionStatus(status="complete")

    content_dicts = [c.model_dump(by_alias=True, exclude_none=True) for c in content]
    (out_dir / "content.json").write_text(json.dumps(content_dicts), encoding="utf-8")

    report = TranscriptionReport(
        status="complete",
        expected=1,
        run_id="test-run",
        identity=TranscriptionIdentity(provider="mock", model="one", prompt_version=1),
        captures=[p.model_copy() for p in metadata.pages],
        content_hash=canonical_json_digest(content_dicts),
        pages=[
            TranscriptionReportPage(
                index=0,
                status="success",
                hash=digest(b"page-image"),
            )
        ],
    )
    (out_dir / "transcription.json").write_text(
        json.dumps(report.model_dump(by_alias=True)), encoding="utf-8"
    )

    assert_complete_transcription(out_dir, metadata, content)

    # Tampered content
    tampered_chunk = ContentChunk(
        index=content[0].index,
        page=content[0].page,
        screenshot=content[0].screenshot,
        text="changed",
    )
    with pytest.raises(ValueError, match="completion report"):
        assert_complete_transcription(out_dir, metadata, [tampered_chunk])

    # Tampered screenshot file
    Path(metadata.pages[0].screenshot).write_bytes(b"changed image")
    with pytest.raises(ValueError, match="screenshot changed"):
        assert_complete_transcription(out_dir, metadata, content)


def test_rejects_running_report_even_when_old_text_remains(book_fixture) -> None:
    out_dir, metadata, content = book_fixture
    report_dict = {
        "status": "running",
        "expected": 1,
        "runId": "run-1",
        "identity": {"provider": "m", "model": "o", "promptVersion": 1, "temperature": 0.0},
        "captures": [p.model_dump(by_alias=True) for p in metadata.pages],
        "pages": [],
    }
    (out_dir / "transcription.json").write_text(json.dumps(report_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="unfinished"):
        assert_complete_transcription(out_dir, metadata, content)


def test_try_read_json_file_handles_missing_file(tmp_path: Path) -> None:
    assert try_read_json_file(tmp_path / "missing.json") is None
