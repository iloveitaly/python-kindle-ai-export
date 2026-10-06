"""Kindle AI Export - Python package."""

import os
from .cli import cli, main
from .models import (
    BookChapter,
    BookMetadata,
    ContentChunk,
    ExtractionStatus,
    PageChunk,
    PageNav,
    TocItem,
)
from .version import __version__

# Suppress pydantic-ai banner unless user requested otherwise
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

__all__ = [
    "BookChapter",
    "BookMetadata",
    "ContentChunk",
    "ExtractionStatus",
    "PageChunk",
    "PageNav",
    "TocItem",
    "__version__",
    "cli",
    "main",
]
