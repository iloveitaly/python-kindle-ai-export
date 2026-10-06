"""Deterministic and filesystem-safe filename generation for exported books."""

import re
import unicodedata

from .models import AmazonBookMeta
from .utils import get_env

CONTROL_OR_ILLEGAL_REGEX = re.compile(r'[\x00-\x1f\x7f-\x9f<>:"/\\|?*]')
WHITESPACE_REGEX = re.compile(r"\s+")
TRIM_DOTS_SPACES_REGEX = re.compile(r"^[. ]+|[. ]+$")
WINDOWS_RESERVED_REGEX = re.compile(
    r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)", re.IGNORECASE
)


def sanitize(value: str) -> str:
    """Normalize Unicode and replace characters illegal in cross-platform filenames."""
    normalized = unicodedata.normalize("NFC", value)
    cleaned = CONTROL_OR_ILLEGAL_REGEX.sub("_", normalized)
    collapsed = WHITESPACE_REGEX.sub(" ", cleaned)
    return TRIM_DOTS_SPACES_REGEX.sub("", collapsed)


def book_filename(
    meta: AmazonBookMeta,
    override: str | None = None,
) -> str:
    """Construct a sanitized filename including author, title, and ASIN disambiguator."""
    active_override = get_env("OUTPUT_BASENAME") if override is None else override
    suffix = "" if active_override else f" [{sanitize(meta.asin)[:40] or 'unknown'}]"

    authors_str = ", ".join(meta.author_list) or "Unknown author"
    title_str = meta.title or meta.asin
    source = active_override or f"{authors_str} - {title_str}"

    name = sanitize(source) or "book"
    while len((name + suffix).encode("utf-8")) > 200:
        name = name[:-1]

    if WINDOWS_RESERVED_REGEX.match(name):
        name = f"_{name}"

    return name + suffix
