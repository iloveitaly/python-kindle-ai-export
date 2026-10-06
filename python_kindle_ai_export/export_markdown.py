"""Export transcribed book content to Markdown with structured chapters and TOC."""

from pathlib import Path

from .book_chapters import book_chapters
from .book_data import (
    assert_complete_extraction,
    assert_complete_transcription,
    read_book_content,
    read_book_metadata,
)
from .book_filename import book_filename
from .utils import atomic_write, get_env


def export_markdown(asin: str | None = None, out_dir: Path | None = None) -> Path:
    """Generate Markdown export for a book from its metadata and transcription content."""
    active_asin = asin or get_env("ASIN")
    if out_dir is not None:
        base_dir = out_dir
    elif active_asin:
        base_dir = Path("out") / active_asin
    else:
        raise ValueError("ASIN is required")
    metadata_path = base_dir / "metadata.json"
    content_path = base_dir / "content.json"

    metadata = read_book_metadata(metadata_path)
    content = read_book_content(content_path)
    assert_complete_extraction(metadata)
    assert_complete_transcription(base_dir, metadata, content)

    chapters = book_chapters(content, metadata.toc)
    toc_lines = [f"- [{ch.label}](#chapter-{i + 1})" for i, ch in enumerate(chapters)]
    toc_str = "\n".join(toc_lines)

    authors_str = ", ".join(metadata.meta.author_list) or "Unknown author"
    doc_lines: list[str] = [
        f"# {metadata.meta.title}",
        "",
        f"> By {authors_str}",
        "",
        "## Table of Contents",
        "",
        toc_str,
    ]

    for i, chapter in enumerate(chapters):
        heading_prefix = "#" * min(6, chapter.depth + 2)
        body = "\n\n".join(chunk.text for chunk in chapter.chunks)
        doc_lines.extend(
            [
                "",
                f'<a id="chapter-{i + 1}"></a>',
                "",
                f"{heading_prefix} {chapter.label}",
                "",
                body,
            ]
        )

    output_text = "\n".join(doc_lines)
    dest_path = base_dir / f"{book_filename(metadata.meta)}.md"
    atomic_write(dest_path, output_text)
    return dest_path
