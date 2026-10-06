"""Tests for chapter segmentation, filename sanitization, and audio batching."""

from python_kindle_ai_export.book_chapters import (
    audio_batches,
    book_chapters,
    narration_content,
)
from python_kindle_ai_export.book_filename import book_filename
from python_kindle_ai_export.models import AmazonBookMeta, ContentChunk, TocItem


def make_chunk(index: int, **fields) -> ContentChunk:
    return ContentChunk(
        index=index,
        screenshot=f"{index}.png",
        text=f"Text {index}",
        **fields,
    )


def make_toc(label: str, **fields) -> TocItem:
    params = {"position_id": -1, "depth": 0, **fields}
    return TocItem(
        label=label,
        **params,
    )


def test_chapter_coverage_retains_front_matter_and_last_chapter_once() -> None:
    content = [
        make_chunk(0, roman_page=1),
        make_chunk(1, page=1),
        make_chunk(2, page=2),
        make_chunk(3, page=3),
    ]
    result = book_chapters(
        content,
        [make_toc("First", page=1), make_toc("Last", page=3)],
    )
    assert [ch.label for ch in result] == ["Opening content", "First", "Last"]
    all_chunks = [c for ch in result for c in ch.chunks]
    assert all_chunks == content


def test_preserves_content_without_mapped_boundaries_and_does_not_mix_locations_with_pages() -> None:
    content = [make_chunk(0, location=2), make_chunk(1, location=3)]
    res = book_chapters(content, [make_toc("Page", page=2)])
    assert len(res) == 1
    assert res[0].label == "Content"
    assert res[0].chunks == content


def test_combines_nested_duplicate_boundaries_without_duplicating() -> None:
    content = [
        make_chunk(0, position=10, end_position=20),
        make_chunk(1, position=20, end_position=30),
    ]
    res = book_chapters(
        content,
        [
            make_toc("Part", position_id=10),
            make_toc("Chapter", position_id=12, depth=1),
            make_toc("Missing", position_id=99),
        ],
    )
    assert len(res) == 1
    assert res[0].label == "Part / Chapter"
    assert res[0].chunks == content


def test_does_not_override_contradictory_position_intervals_with_printed_pages() -> None:
    content = [make_chunk(0, page=1, position=100, end_position=200)]
    res = book_chapters(content, [make_toc("Incorrect", page=1, position_id=10)])
    assert res[0].label == "Content"


def test_narration_preview_first_capture_independently_of_page_number() -> None:
    content = [make_chunk(1, page=1), make_chunk(0, roman_page=1)]
    preview = narration_content(content, preview=True)
    assert [item.index for item in preview] == [0]

    batches = [
        batch
        for ch in book_chapters(preview, [])
        for batch in audio_batches(" ".join(c.text for c in ch.chunks), 100)
    ]
    assert batches == ["Text 0"]

    full = narration_content(content, preview=False)
    assert [item.index for item in full] == [0, 1]


def test_uses_capture_order_and_preserves_repeated_page_screenshots() -> None:
    content = [
        make_chunk(1, page=1),
        make_chunk(0, page=1),
        make_chunk(2, page=2),
    ]
    chapters = book_chapters(content, [make_toc("Start", page=1)])
    assert [c.index for ch in chapters for c in ch.chunks] == [0, 1, 2]


def test_book_filename_preserves_unicode_and_prevents_collisions() -> None:
    meta = AmazonBookMeta(
        asin="B123",
        title="日本語: A/B",
        author_list=["Zoë"],
    )
    assert book_filename(meta, "") == "Zoë - 日本語_ A_B [B123]"
    assert "/" not in book_filename(meta, "../../book")
    assert book_filename(meta, "book") == "book"
    assert book_filename(meta, "CON") == "_CON"

    long_meta = AmazonBookMeta(
        asin="B123",
        title="語" * 500,
        author_list=["Zoë"],
    )
    assert len(book_filename(long_meta, "").encode("utf-8")) <= 200


def test_audio_batches_splits_oversized_paragraphs() -> None:
    text = ("word " * 2000).strip()
    batches = audio_batches(text, 100)
    assert all(len(b) <= 100 for b in batches)
    assert " ".join(batches) == text

    short_batches = audio_batches("x" * 201, 100)
    assert [len(b) for b in short_batches] == [100, 100, 1]
