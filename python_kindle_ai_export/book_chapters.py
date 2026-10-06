"""Book chapter segmentation, narration content filtering, and audio batching."""

from .models import BookChapter, ContentChunk, TocItem


def book_chapters(
    content: list[ContentChunk],
    toc: list[TocItem],
) -> list[BookChapter]:
    """Segment ordered captures into chapters based on verified positions and TOC entries.

    Non-obvious behavior and design decisions:
    1. Multi-Coordinate Hierarchy: Kindle books lack a standardized coordinate system across formats.
       The matching cascade evaluates:
       - Internal Position Range: ``chunk.position <= item.position_id < chunk.end_position``
       - Fallback Coordinates: Checks ``roman_page``, ``page``, and ``location`` in order.
    2. Opening Content Synthesis: Chunks appearing before the first detected TOC boundary are
       preserved as a synthetic "Opening content" chapter (preface, title page, copyright, etc.).
    3. Coincident Boundary Merging: When multiple TOC entries start on the same screen (e.g.
       a "Part I" section header and "Chapter 1" on the same page), their labels are joined
       as ``"Part I / Chapter 1"`` and assigned the minimum depth level.
    """
    ordered = sorted(content, key=lambda c: c.index)
    boundaries: dict[int, list[TocItem]] = {}

    for item in toc:
        index: int = -1
        for i, chunk in enumerate(ordered):
            if chunk.position is not None and (
                chunk.position == item.position_id
                or (
                    chunk.end_position is not None
                    and chunk.position <= item.position_id < chunk.end_position
                )
            ):
                index = i
                break

        if index < 0:
            for coordinate in ("roman_page", "page", "location"):
                target_val = getattr(item, coordinate)
                if target_val is None:
                    continue

                for i, chunk in enumerate(ordered):
                    if getattr(chunk, coordinate) == target_val and (
                        chunk.position is None
                        or (
                            chunk.position <= item.position_id
                            and (
                                chunk.end_position is None
                                or item.position_id < chunk.end_position
                            )
                        )
                    ):
                        index = i
                        break

                if index >= 0:
                    break

        if index < 0:
            continue

        boundaries.setdefault(index, []).append(item)

    if not ordered:
        return []

    if not boundaries:
        return [BookChapter(label="Content", depth=0, chunks=ordered)]

    starts = sorted(boundaries.keys())
    chapters: list[BookChapter] = []
    if starts[0] > 0:
        chapters.append(
            BookChapter(label="Opening content", depth=0, chunks=ordered[: starts[0]])
        )

    for i, start in enumerate(starts):
        items = boundaries[start]
        next_start = starts[i + 1] if i + 1 < len(starts) else len(ordered)
        unique_labels = list(dict.fromkeys(item.label for item in items))
        label = " / ".join(unique_labels)
        depth = min(item.depth for item in items)
        chapters.append(
            BookChapter(
                label=label,
                depth=depth,
                chunks=ordered[start:next_start],
            )
        )

    return chapters


def audio_batches(text: str, limit: int) -> list[str]:
    """Split text into word-bounded segments with bounded length for TTS requests.

    Non-obvious behavior:
    Finds the last space before ``limit``. If no space exists in the second half of the
    window (``boundary > limit // 2``), it hard-splits at ``limit`` to prevent pathological
    under-sized batches when dealing with long URLs or unspaced punctuation sequences.
    """
    batches: list[str] = []
    remaining = text.strip()
    while len(remaining) > limit:
        boundary = remaining.rfind(" ", 0, limit)
        end = boundary if boundary > limit // 2 else limit
        batches.append(remaining[:end])
        remaining = remaining[end:].lstrip()

    if remaining:
        batches.append(remaining)

    return batches


def narration_content(
    content: list[ContentChunk],
    preview: bool,
) -> list[ContentChunk]:
    """Return captures for narration; preview returns only the first capture."""
    ordered = sorted(content, key=lambda c: c.index)
    return ordered[:1] if preview else ordered
