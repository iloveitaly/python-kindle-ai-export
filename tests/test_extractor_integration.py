"""Integration tests for the Kindle extractor with a simulated browser driver."""

import asyncio
import io
import json
import signal
import tarfile
from pathlib import Path
from typing import Callable
from unittest.mock import AsyncMock, MagicMock
import pytest
from python_kindle_ai_export.extractor import extract_kindle_book
from python_kindle_ai_export.models import BookMetadata
from python_kindle_ai_export.pacing import PacingConfig


def create_simulated_browser(
    stuck: bool = False,
    initial: int = 1,
    cancel: bool = False,
    metadata_responses: bool = False,
):
    current = initial
    cancelled = False
    late_metadata_sent = False
    response_handlers: list[Callable] = []
    destination = "1"
    screens = ["Page i of 1", "Page 1 of 1", "Page 1 of 1"]

    def is_disabled(sel: str) -> bool:
        if "chevron-container-left" in sel:
            return current == 0
        if "chevron-container-right" in sel:
            return current == 2
        return False

    def locator(selector: str):
        mock_el = MagicMock()
        mock_el.first = mock_el
        mock_el.filter.return_value = mock_el
        mock_el.wait_for = AsyncMock()
        mock_el.hover = AsyncMock()

        async def fill(val: str) -> None:
            nonlocal destination
            destination = val

        mock_el.fill = AsyncMock(side_effect=fill)

        async def click(**kwargs) -> None:
            nonlocal current, cancelled, late_metadata_sent
            if "go-to-modal-go-button" in selector:
                current = 0 if destination == "i" else 1

            if "chevron-container-left" in selector and current > 0:
                current -= 1

            if "chevron-container-right" in selector and current < 2 and not (stuck and current == 1):
                current += 1

            if "chevron-container-right" in selector and metadata_responses and not late_metadata_sent:
                late_metadata_sent = True
                mock_resp = MagicMock()
                mock_resp.status = 200
                mock_resp.url = "https://read.amazon.com/YJmetadata.jsonp"
                mock_resp.text = AsyncMock(
                    return_value='callback({"asin":"TEST","title":"Late network title","authorList":["First Author","Second Author"]})'
                )
                for h in response_handlers:
                    h(mock_resp)

            if "chevron-container-right" in selector and cancel and not cancelled:
                cancelled = True
                loop = asyncio.get_running_loop()
                # Trigger SIGINT
                for sig in (signal.SIGINT,):
                    try:
                        loop._process_signal(sig)  # pyright: ignore[reportAttributeAccessIssue]
                    except Exception:
                        pass

        mock_el.click = AsyncMock(side_effect=click)

        async def is_visible() -> bool:
            return "ion-alert" not in selector

        mock_el.is_visible = AsyncMock(side_effect=is_visible)

        async def is_enabled() -> bool:
            return not is_disabled(selector)

        mock_el.is_enabled = AsyncMock(side_effect=is_enabled)

        async def get_attribute(name: str) -> str | None:
            if name == "src":
                return f"blob:screen-{current}"
            return "true" if is_disabled(selector) else None

        mock_el.get_attribute = AsyncMock(side_effect=get_attribute)
        mock_el.text_content = AsyncMock(side_effect=lambda **kw: screens[current])
        mock_el.all_text_contents = AsyncMock(return_value=[])
        mock_el.screenshot = AsyncMock(side_effect=lambda **kw: f"screen-{current}".encode())

        async def evaluate(fn_or_str: str) -> object:
            if "naturalWidth" in fn_or_str:
                return True
            if "aria-disabled" in fn_or_str:
                return is_disabled(selector)
            return True

        mock_el.evaluate = AsyncMock(side_effect=evaluate)
        return mock_el

    page = MagicMock()
    page.locator = MagicMock(side_effect=locator)
    page.route = AsyncMock()
    page.expose_binding = AsyncMock()
    page.on = MagicMock(side_effect=lambda event, h: response_handlers.append(h) if event == "response" else None)
    page.keyboard = MagicMock()
    page.keyboard.press = AsyncMock()
    page.url = "https://read.amazon.com/?asin=TEST"

    def make_tar() -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w"):
            pass
        return buf.getvalue()

    async def goto(url: str, **kwargs) -> None:
        if metadata_responses:
            r1 = MagicMock()
            r1.status = 200
            r1.url = "https://read.amazon.com/YJmetadata.jsonp"
            r1.text = AsyncMock(
                return_value='callback({"asin":"TEST","title":"Network title","authorList":["First Author","Second Author"]})'
            )
            for h in response_handlers:
                h(r1)

            r2 = MagicMock()
            r2.status = 200
            r2.url = "https://read.amazon.com/renderer/render?asin=TEST"
            r2.body = AsyncMock(return_value=make_tar())
            for h in response_handlers:
                h(r2)

    page.goto = AsyncMock(side_effect=goto)

    prod_page = MagicMock()
    prod_page.goto = AsyncMock(side_effect=Exception("Product unavailable"))
    prod_page.close = AsyncMock()

    context = MagicMock()
    context.pages = [page]
    context.add_init_script = AsyncMock()
    context.new_page = AsyncMock(return_value=prod_page)
    context.request = MagicMock()
    context.request.get = AsyncMock(return_value=MagicMock(ok=False))
    context.close = AsyncMock()

    return context, lambda: current


@pytest.mark.asyncio
async def test_extractor_rewinds_captures_all_and_restores(tmp_path: Path) -> None:
    context, get_pos = create_simulated_browser(stuck=False, initial=2)
    out_dir = tmp_path / "out" / "TEST"

    meta = await extract_kindle_book(
        asin="TEST",
        amazon_email="test@example.com",
        amazon_password="test-password",
        out_dir=out_dir,
        context=context,
        max_screens=10,
        sleep_fn=AsyncMock(),
        pacing=PacingConfig(enabled=False),
    )

    # Verify pages: 0 (roman 1), 1 (page 1), 2 (page 1)
    coords = [(p.index, p.roman_page, p.page) for p in meta.pages]
    assert coords == [
        (0, 1, None),
        (1, None, 1),
        (2, None, 1),
    ]
    assert meta.extraction is not None
    assert meta.extraction.status == "complete"
    assert meta.extraction.restoration == "succeeded"
    assert get_pos() == 2


@pytest.mark.asyncio
async def test_reports_stalled_middle_screen_as_partial(tmp_path: Path) -> None:
    context, get_pos = create_simulated_browser(stuck=True, initial=1)
    out_dir = tmp_path / "out" / "TEST_STUCK"

    meta = await extract_kindle_book(
        asin="TEST_STUCK",
        amazon_email="test@example.com",
        amazon_password="test-password",
        out_dir=out_dir,
        context=context,
        max_screens=10,
        sleep_fn=AsyncMock(),
        pacing=PacingConfig(enabled=False),
    )

    assert len(meta.pages) == 2
    assert meta.extraction is not None
    assert meta.extraction.status == "partial"
    assert meta.extraction.restoration == "succeeded"
    assert get_pos() == 1


@pytest.mark.asyncio
async def test_drains_network_metadata_and_tolerates_missing_render_files(tmp_path: Path) -> None:
    context, _ = create_simulated_browser(stuck=False, initial=1, metadata_responses=True)
    out_dir = tmp_path / "out" / "TEST_META"

    meta = await extract_kindle_book(
        asin="TEST",
        amazon_email="test@example.com",
        amazon_password="test-password",
        out_dir=out_dir,
        context=context,
        max_screens=10,
        sleep_fn=AsyncMock(),
        pacing=PacingConfig(enabled=False),
    )

    assert meta.meta.title == "Late network title"
    assert meta.meta.author_list == ["First Author", "Second Author"]
    assert meta.location_map is None
    assert meta.toc == []
    assert meta.extraction is not None
    assert meta.extraction.status == "complete"


@pytest.mark.asyncio
async def test_extractor_takes_pacing_breaks(tmp_path: Path) -> None:
    context, _ = create_simulated_browser(stuck=False, initial=2)
    sleep_mock = AsyncMock()

    meta = await extract_kindle_book(
        asin="TEST",
        amazon_email="test@example.com",
        amazon_password="test-password",
        out_dir=tmp_path / "out" / "TEST_PACED",
        context=context,
        max_screens=10,
        sleep_fn=sleep_mock,
        pacing=PacingConfig(
            action_delay=(0.0, 0.0),
            window_seconds=10_000,
            max_actions_per_window=(2, 2),
            break_seconds=(3.0, 3.0),
        ),
    )

    assert meta.extraction is not None
    assert meta.extraction.status == "complete"
    # break slices are exactly 1.0s; the extractor's own polling sleeps are 0.2s / 0.5s
    break_slices = [c for c in sleep_mock.await_args_list if c.args == (1.0,)]
    assert len(break_slices) >= 3
    assert len(break_slices) % 3 == 0


def _redirect_to_signin(context: MagicMock) -> MagicMock:
    page = context.pages[0]
    page.logged_in = False
    original_goto = page.goto

    async def goto(url: str, **kwargs) -> None:
        await original_goto(url, **kwargs)
        if not page.logged_in:
            page.url = "https://www.amazon.com/ap/signin?openid.return_to=read.amazon.com"
        else:
            page.url = url

    page.goto = AsyncMock(side_effect=goto)
    return page


@pytest.mark.asyncio
async def test_manual_login_waits_for_user_then_continues(tmp_path: Path) -> None:
    context, _ = create_simulated_browser(stuck=False, initial=2)
    page = _redirect_to_signin(context)
    login_polls = 0

    async def fake_sleep(seconds: float) -> None:
        nonlocal login_polls
        if seconds == 2.0:
            login_polls += 1
            if login_polls == 3:  # user finishes sign-in + 2FA
                page.logged_in = True
                page.url = "https://read.amazon.com/kindle-library"

    meta = await extract_kindle_book(
        asin="TEST",
        out_dir=tmp_path / "out" / "TEST_MANUAL",
        context=context,
        max_screens=10,
        sleep_fn=AsyncMock(side_effect=fake_sleep),
        pacing=PacingConfig(enabled=False),
    )

    assert login_polls == 3
    assert meta.extraction is not None
    assert meta.extraction.status == "complete"
    # never touched the sign-in form
    filled = [c for c in page.locator.call_args_list if "input[type=" in str(c)]
    assert filled == []


@pytest.mark.asyncio
async def test_manual_login_times_out(tmp_path: Path) -> None:
    context, _ = create_simulated_browser(stuck=False, initial=2)
    _redirect_to_signin(context)

    meta = await extract_kindle_book(
        asin="TEST",
        out_dir=tmp_path / "out" / "TEST_MANUAL_TIMEOUT",
        context=context,
        max_screens=10,
        sleep_fn=AsyncMock(),
        pacing=PacingConfig(enabled=False),
        manual_login=True,
        login_timeout_seconds=10,
    )

    assert meta.extraction is not None
    assert meta.extraction.status == "partial"
    assert meta.extraction.reason == "Timed out waiting for manual login"
    assert meta.pages == []


@pytest.mark.asyncio
async def test_auto_login_requires_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AMAZON_EMAIL", "AMAZON_PASSWORD"):
        for prefix in ("", "KINDLE_EXPORT_", "KINDLE_AI_EXPORT_"):
            monkeypatch.delenv(f"{prefix}{name}", raising=False)
    context, _ = create_simulated_browser()
    with pytest.raises(ValueError, match="required unless manual login"):
        await extract_kindle_book(
            asin="TEST",
            out_dir=tmp_path / "out" / "X",
            context=context,
            sleep_fn=AsyncMock(),
            pacing=PacingConfig(enabled=False),
            manual_login=False,
        )


@pytest.mark.asyncio
async def test_extractor_resumes_from_existing_checkpoint(tmp_path: Path) -> None:
    context1, _ = create_simulated_browser(stuck=False, initial=2)
    out_dir = tmp_path / "out" / "TEST_RESUME"

    # Pass 1: Capture only the first screen (index 0)
    meta1 = await extract_kindle_book(
        asin="TEST",
        amazon_email="test@example.com",
        amazon_password="test-password",
        out_dir=out_dir,
        context=context1,
        max_screens=1,
        sleep_fn=AsyncMock(),
        pacing=PacingConfig(enabled=False),
    )

    assert len(meta1.pages) == 1
    assert meta1.pages[0].index == 0
    assert meta1.pages[0].roman_page == 1
    assert meta1.extraction is not None
    assert meta1.extraction.status == "partial"

    # Pass 2: Resume with max_screens=2 to capture the remaining screens
    context2, _ = create_simulated_browser(stuck=False, initial=0)
    meta2 = await extract_kindle_book(
        asin="TEST",
        amazon_email="test@example.com",
        amazon_password="test-password",
        out_dir=out_dir,
        context=context2,
        max_screens=5,
        sleep_fn=AsyncMock(),
        pacing=PacingConfig(enabled=False),
        resume=True,
    )

    # Combined pages: 0, 1, 2
    assert len(meta2.pages) == 3
    indices = [p.index for p in meta2.pages]
    assert indices == [0, 1, 2]
    assert meta2.extraction is not None
    assert meta2.extraction.status == "complete"
    assert meta2.extraction.restoration == "succeeded"


@pytest.mark.asyncio
async def test_extractor_returns_existing_when_already_complete(tmp_path: Path) -> None:
    context1, _ = create_simulated_browser(stuck=False, initial=2)
    out_dir = tmp_path / "out" / "TEST_ALREADY_COMPLETE"

    # Complete extraction
    meta1 = await extract_kindle_book(
        asin="TEST",
        amazon_email="test@example.com",
        amazon_password="test-password",
        out_dir=out_dir,
        context=context1,
        max_screens=10,
        sleep_fn=AsyncMock(),
        pacing=PacingConfig(enabled=False),
    )
    assert meta1.extraction is not None
    assert meta1.extraction.status == "complete"

    # Second invocation with resume=True should return existing without re-running
    context2, _ = create_simulated_browser(stuck=False, initial=2)
    context2.pages[0].goto = AsyncMock()
    meta2 = await extract_kindle_book(
        asin="TEST",
        amazon_email="test@example.com",
        amazon_password="test-password",
        out_dir=out_dir,
        context=context2,
        max_screens=10,
        sleep_fn=AsyncMock(),
        pacing=PacingConfig(enabled=False),
        resume=True,
    )
    assert len(meta2.pages) == 3
    assert meta2.extraction is not None
    assert meta2.extraction.status == "complete"
    # Never navigated
    context2.pages[0].goto.assert_not_called()
