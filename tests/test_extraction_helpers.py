"""Tests for reader coordinates, domain parsing, and boundary observation helpers."""

import pytest
from python_kindle_ai_export.extraction_helpers import (
    BoundarySnapshot,
    confirmed_boundary,
    inspect_boundary,
    parse_page_label,
    parse_page_nav,
    reader_domain,
)
from python_kindle_ai_export.models import PageNav


def test_reader_coordinates_roman_vs_location() -> None:
    assert parse_page_nav("Page xiv of 230") == PageNav(
        roman_page=14,
        label="xiv",
        total=230,
    )
    assert parse_page_nav("Location 14 of 2,300") == PageNav(
        location=14,
        label="14",
        total=2300,
    )
    assert parse_page_nav("Page 14 of 230") == PageNav(
        page=14,
        label="14",
        total=230,
    )


def test_unknown_map_labels_do_not_invent_pages() -> None:
    assert parse_page_label("Cover") == {}
    assert parse_page_label("12a") == {}
    assert parse_page_nav("Loading") is None


def test_reader_domain_regional_hosts() -> None:
    assert reader_domain("https://read.amazon.co.jp/") == "amazon.co.jp"
    with pytest.raises(ValueError, match="supported Amazon regional domain"):
        reader_domain("amazon.com.evil.test")


def test_confirmed_boundary_requires_visible_disabled() -> None:
    assert confirmed_boundary(True, True) is True
    assert confirmed_boundary(False, True) is False
    assert confirmed_boundary(True, False) is False
    assert confirmed_boundary(False, False) is False


@pytest.mark.asyncio
async def test_boundary_inspection_recognizes_stable_removed_control_after_failure() -> None:
    async def pause() -> None:
        pass

    loaded = BoundarySnapshot(
        key="screen-1",
        loaded=True,
        visible=False,
        disabled=False,
        opposite_enabled=True,
    )

    assert await inspect_boundary(lambda: async_return(loaded), pause) is False
    assert await inspect_boundary(lambda: async_return(loaded), pause, after_failed_navigation=True) is True


@pytest.mark.asyncio
async def test_boundary_inspection_rejects_transient_or_hidden() -> None:
    async def pause() -> None:
        pass

    calls = 0

    async def changing_key() -> BoundarySnapshot:
        nonlocal calls
        calls += 1
        return BoundarySnapshot(
            key=str(calls),
            loaded=True,
            visible=False,
            disabled=False,
            opposite_enabled=True,
        )

    assert await inspect_boundary(changing_key, pause, after_failed_navigation=True) is False

    not_loaded = BoundarySnapshot(
        key="screen-1",
        loaded=False,
        visible=False,
        disabled=False,
        opposite_enabled=True,
    )
    assert await inspect_boundary(lambda: async_return(not_loaded), pause, after_failed_navigation=True) is False

    opposite_disabled = BoundarySnapshot(
        key="screen-1",
        loaded=True,
        visible=False,
        disabled=False,
        opposite_enabled=False,
    )
    assert await inspect_boundary(lambda: async_return(opposite_disabled), pause, after_failed_navigation=True) is False


async def async_return[T](val: T) -> T:
    return val
