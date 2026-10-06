"""Export book content or captured images to PDF."""

import logging
import sys
import uuid
from pathlib import Path
from typing import Literal
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from .book_chapters import book_chapters
from .book_data import (
    assert_complete_extraction,
    assert_complete_transcription,
    read_book_content,
    read_book_metadata,
    resolve_screenshot_path,
)
from .book_filename import book_filename
from .log import log
from .models import ContentChunk
from .utils import get_env


def export_pdf(
    asin: str | None = None,
    out_dir: Path | None = None,
    mode: str | None = None,
) -> Path:
    """Generate a PDF document from captured images or transcribed text."""
    active_asin = asin or get_env("ASIN")
    if out_dir is not None:
        base_dir = out_dir
    elif active_asin:
        base_dir = Path("out") / active_asin
    else:
        raise ValueError("ASIN is required")

    active_mode = (mode or get_env("PDF_MODE") or "auto").lower()
    if active_mode not in ("auto", "text", "images"):
        raise ValueError("PDF_MODE must be auto, text, or images")

    metadata = read_book_metadata(base_dir / "metadata.json")
    assert_complete_extraction(metadata)

    content: list[ContentChunk] | None = None
    if active_mode != "images":
        try:
            content = read_book_content(base_dir / "content.json")
            assert_complete_transcription(base_dir, metadata, content)
        except Exception as err:
            if active_mode == "text":
                raise
            content = None
            sys.stderr.write(
                f"Text transcription unavailable or incomplete; exporting captured images: {err}\n"
            )

    if not content and not metadata.pages:
        raise ValueError("no book content found")

    dest_path = base_dir / f"{book_filename(metadata.meta)}.pdf"
    temp_path = dest_path.with_name(f"{dest_path.name}.{uuid.uuid4()}.tmp")

    c = canvas.Canvas(str(temp_path))
    c.setTitle(metadata.meta.title)
    authors_str = ", ".join(metadata.meta.author_list) or "Unknown author"
    c.setAuthor(authors_str)

    try:
        if not content:
            sorted_pages = sorted(metadata.pages, key=lambda p: p.index)
            for page in sorted_pages:
                img_path = resolve_screenshot_path(base_dir, page.screenshot)
                with Image.open(img_path) as img:
                    w, h = img.size
                c.setPageSize((float(w), float(h)))
                c.drawImage(str(img_path), 0, 0, width=float(w), height=float(h))
                c.showPage()
        else:
            page_w, page_h = letter
            c.setPageSize((page_w, page_h))

            cover_rendered = False
            if metadata.meta.cover_path:
                cover_path = resolve_screenshot_path(base_dir, metadata.meta.cover_path)
                if cover_path.is_file():
                    try:
                        with Image.open(cover_path) as c_img:
                            cw, ch = c_img.size
                        ratio = min((page_w - 100) / cw, (page_h - 100) / ch)
                        draw_w, draw_h = cw * ratio, ch * ratio
                        c.drawImage(
                            str(cover_path),
                            (page_w - draw_w) / 2,
                            (page_h - draw_h) / 2,
                            width=draw_w,
                            height=draw_h,
                        )
                        cover_rendered = True
                    except Exception as err:
                        log.warning(
                            "could not render cover; using title page",
                            error=str(err),
                        )

            if not cover_rendered:
                c.setFont("Helvetica-Bold", 24)
                c.drawCentredString(page_w / 2, page_h - 200, metadata.meta.title)
                c.setFont("Helvetica", 16)
                c.drawCentredString(page_w / 2, page_h - 240, f"By {authors_str}")

            c.showPage()

            chapters = book_chapters(content, metadata.toc)
            for i, chapter in enumerate(chapters):
                c.setPageSize((page_w, page_h))
                bookmark_name = f"ch_{i + 1}"
                c.bookmarkPage(bookmark_name)
                c.addOutlineEntry(chapter.label, bookmark_name, level=min(chapter.depth, 2))

                c.setFont("Helvetica-Bold", 18)
                c.drawCentredString(page_w / 2, page_h - 72, chapter.label)

                text_obj = c.beginText(54, page_h - 110)
                text_obj.setFont("Helvetica", 11)
                text_obj.setLeading(15)

                for chunk in chapter.chunks:
                    for line in chunk.text.split("\n"):
                        text_obj.textLine(line)
                    text_obj.textLine("")

                c.drawText(text_obj)
                c.showPage()

        c.save()
        temp_path.replace(dest_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return dest_path
