"""Helpers for reader coordinates, domain validation, and navigation boundary detection."""

import re
from collections.abc import Awaitable, Callable
from pydantic import BaseModel

from .models import PageNav
from .utils import deromanize

PAGE_LABEL_DIGITS_REGEX = re.compile(r"^\d+$")
PAGE_LABEL_ROMAN_REGEX = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
SUPPORTED_DOMAIN_REGEX = re.compile(
    r"^(?:read\.)?amazon\.(?:com|co\.uk|co\.jp|com\.au|com\.br|com\.mx|ca|de|fr|it|es|in|nl)$"
)
PAGE_NAV_REGEX = re.compile(
    r"(page|location|loc\.?)\s+([\divxlcdm,]+)\s+(?:of|\/)\s+([\d,]+)",
    re.IGNORECASE,
)


class BoundarySnapshot(BaseModel):
    """Observation of reader navigation controls and page state."""

    key: str
    loaded: bool
    visible: bool
    disabled: bool
    opposite_enabled: bool


def parse_page_label(label: str) -> dict[str, int]:
    """Extract numeric or Roman page coordinates from a page label."""
    if PAGE_LABEL_DIGITS_REGEX.match(label):
        return {"page": int(label)}

    if PAGE_LABEL_ROMAN_REGEX.match(label):
        roman_page = deromanize(label)
        if roman_page > 0:
            return {"roman_page": roman_page}

    return {}


def parse_page_nav(text: str | None) -> PageNav | None:
    """Parse footer navigation text such as 'Page 14 of 230' or 'Location 14 of 2,300'."""
    if not text:
        return None

    match = PAGE_NAV_REGEX.search(text)
    if not match:
        return None

    kind = match.group(1).lower()
    label = match.group(2).replace(",", "")
    total_str = match.group(3).replace(",", "")
    if not total_str.isdigit():
        return None

    total = int(total_str)
    if total <= 0:
        return None

    if kind == "page":
        coords = parse_page_label(label)
        if "page" not in coords and "roman_page" not in coords:
            return None
        return PageNav(
            page=coords.get("page"),
            roman_page=coords.get("roman_page"),
            label=label,
            total=total,
        )

    try:
        location = int(label)
        return PageNav(location=location, label=label, total=total)
    except ValueError:
        return None


def reader_domain(value: str = "amazon.com") -> str:
    """Normalize and validate an Amazon regional host."""
    domain = (
        value.strip()
        .lower()
        .removeprefix("https://")
        .removeprefix("http://")
        .rstrip("/")
    )
    if not SUPPORTED_DOMAIN_REGEX.match(domain):
        raise ValueError("AMAZON_DOMAIN must be a supported Amazon regional domain")

    return domain.removeprefix("read.")


def confirmed_boundary(disabled: bool, visible: bool) -> bool:
    """Confirm whether a navigation boundary is reached by a visible disabled control."""
    return visible and disabled


async def inspect_boundary(
    read_fn: Callable[[], Awaitable[BoundarySnapshot]],
    pause_fn: Callable[[], Awaitable[None]],
    after_failed_navigation: bool = False,
) -> bool:
    """Verify that reader endpoint controls remain reliably disabled or absent over observations."""
    previous_key: str | None = None
    for _ in range(3):
        state = await read_fn()
        if not state.loaded:
            return False

        is_confirmed = confirmed_boundary(state.disabled, state.visible)
        is_stable_removed = (
            after_failed_navigation and not state.visible and state.opposite_enabled
        )
        if not is_confirmed and not is_stable_removed:
            return False

        if previous_key is not None and state.key != previous_key:
            return False

        previous_key = state.key
        await pause_fn()

    return True
