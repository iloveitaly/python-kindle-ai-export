"""Book metadata loading, verification, and completeness checking."""

import json
from pathlib import Path
from typing import TypeVar

from .models import BookMetadata, ContentChunk, ExtractionStatus, PageChunk, TranscriptionReport
from .utils import canonical_json_digest, digest, read_json_file

BOOK_SCHEMA_VERSION = 2

TChunk = TypeVar("TChunk", bound=PageChunk)


def validate_chunks(chunks: list[TChunk]) -> list[TChunk]:
    """Ensure capture chunks have valid, distinct ordered indices and valid coordinates."""
    if not isinstance(chunks, list):
        raise ValueError("book chunks must be an array")

    seen_indices: set[int] = set()
    for chunk in chunks:
        if not isinstance(chunk.index, int) or chunk.index < 0:
            raise ValueError("invalid capture index")

        if chunk.index in seen_indices:
            raise ValueError(f"duplicate capture index: {chunk.index}")

        if not chunk.screenshot:
            raise ValueError("missing screenshot path")

        seen_indices.add(chunk.index)

    return sorted(chunks, key=lambda c: c.index)


def read_book_metadata(file_path: Path) -> BookMetadata:
    """Load and validate book metadata JSON document."""
    raw_data = read_json_file(file_path)
    if not isinstance(raw_data, dict):
        raise ValueError("invalid book metadata")

    schema_version = raw_data.get("schemaVersion", raw_data.get("schema_version"))
    if schema_version not in (None, 1, BOOK_SCHEMA_VERSION):
        raise ValueError("unsupported book metadata version")

    metadata = BookMetadata.model_validate(raw_data)
    metadata.pages = validate_chunks(metadata.pages)

    asin = metadata.meta.asin or file_path.parent.name
    metadata.meta.asin = asin
    metadata.meta.title = metadata.meta.title.strip() if metadata.meta.title else asin
    metadata.meta.author_list = [
        author.strip()
        for author in metadata.meta.author_list
        if isinstance(author, str) and author.strip()
    ]
    return metadata


def read_book_content(file_path: Path) -> list[ContentChunk]:
    """Load and validate transcribed book content chunks."""
    raw_data = read_json_file(file_path)
    if not isinstance(raw_data, list):
        raise ValueError("invalid content format: expected a list of chunks")

    chunks = [ContentChunk.model_validate(item) for item in raw_data]
    validated = validate_chunks(chunks)
    for chunk in validated:
        if not isinstance(chunk.text, str):
            raise ValueError("invalid transcription text")

    return validated


def assert_complete_extraction(metadata: BookMetadata) -> None:
    """Verify that reader extraction reached the end without unconfirmed truncation."""
    if not metadata.pages:
        raise ValueError("no captured pages")

    if metadata.extraction and metadata.extraction.status == "partial":
        reason = metadata.extraction.reason or "end of book was not confirmed"
        raise ValueError(f"extraction is partial: {reason}")

    is_modern_schema = metadata.schema_version and metadata.schema_version >= BOOK_SCHEMA_VERSION
    if is_modern_schema and (not metadata.extraction or metadata.extraction.status != "complete"):
        raise ValueError("extraction completion status is missing")


def resolve_screenshot_path(out_dir: Path, screenshot_path: str) -> Path:
    """Resolve screenshot file path, checking both absolute/cwd and out_dir relative."""
    candidate = Path(screenshot_path)
    if candidate.is_file():
        return candidate
    relative_candidate = out_dir / screenshot_path
    if relative_candidate.is_file():
        return relative_candidate
    return candidate


def assert_complete_transcription(
    out_dir: Path,
    metadata: BookMetadata,
    content: list[ContentChunk],
) -> None:
    """Validate that the published transcript matches its completion report and current images."""
    assert_complete_extraction(metadata)

    if len(content) != len(metadata.pages):
        raise ValueError("transcription is incomplete: capture count differs")

    by_index = {chunk.index: chunk for chunk in content}
    for page in metadata.pages:
        chunk = by_index.get(page.index)
        if not chunk or chunk.screenshot != page.screenshot:
            raise ValueError(f"transcription is missing or stale for capture {page.index}")

        if (
            chunk.page != page.page
            or chunk.roman_page != page.roman_page
            or chunk.location != page.location
            or chunk.position != page.position
            or chunk.end_position != page.end_position
        ):
            raise ValueError(f"transcription coordinates changed for capture {page.index}")

    report_path = out_dir / "transcription.json"
    if not report_path.is_file():
        if (metadata.schema_version or 1) < BOOK_SCHEMA_VERSION:
            return
        raise FileNotFoundError(f"Missing transcription report: {report_path}")

    report_raw = read_json_file(report_path)
    if not isinstance(report_raw, dict):
        raise ValueError("invalid transcription completion report")

    report = TranscriptionReport.model_validate(report_raw)
    if report.status != "complete" or report.expected != len(content):
        raise ValueError("transcription has unfinished or failed pages; rerun transcription")

    content_dicts = [chunk.model_dump(by_alias=True, exclude_none=True) for chunk in content]
    computed_content_hash = canonical_json_digest(content_dicts)
    if report.content_hash != computed_content_hash:
        raise ValueError("transcription content does not match its completion report")

    if len(report.pages) != len(metadata.pages):
        raise ValueError("invalid transcription completion report")

    reported_by_index = {p.index: p for p in report.pages}
    if len(reported_by_index) != len(metadata.pages):
        raise ValueError("duplicate transcription report entries")

    for page in metadata.pages:
        entry = reported_by_index.get(page.index)
        if not entry or entry.status not in ("success", "blank"):
            raise ValueError(f"capture {page.index} is not transcribed")

        image_path = resolve_screenshot_path(out_dir, page.screenshot)
        current_hash = digest(image_path.read_bytes())
        if current_hash != entry.hash:
            raise ValueError(f"screenshot changed for capture {page.index}; rerun transcription")
