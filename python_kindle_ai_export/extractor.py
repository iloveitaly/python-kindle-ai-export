"""Browser-driven reader extraction using camoufox anti-detect automation."""

import asyncio
import base64
from collections.abc import Awaitable, Callable
import io
import json
import logging
import re
import signal
import sys
import tempfile
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from camoufox.async_api import AsyncCamoufox
from PIL import Image
from playwright.async_api import BrowserContext, Page, Request, Response, Route

from .extraction_helpers import (
    BoundarySnapshot,
    inspect_boundary,
    parse_page_label,
    parse_page_nav,
    reader_domain,
)
from .log import log
from .models import (
    AmazonBookMeta,
    AmazonRenderLocationMap,
    AmazonRenderNavigationUnit,
    AmazonRenderTocItem,
    BookMetadata,
    ExtractionStatus,
    Nav,
    PageChunk,
    PageNav,
    TocItem,
)
from .pacing import HumanPacer, PacingConfig
from .utils import (
    atomic_json,
    extract_tar_bytes,
    get_env,
    normalize_authors,
    parse_jsonp_response,
    try_read_json_file,
)

URL_REGEX_BLACKLIST = [
    re.compile(r"unagi-\w+\.amazon\.com", re.IGNORECASE),
    re.compile(r"m\.media-amazon\.com.*\/showads", re.IGNORECASE),
    re.compile(r"fls-na\.amazon\.com.*\/remote-weblab-triggers", re.IGNORECASE),
]

KR_RENDERER_IMAGE_SELECTOR = "#kr-renderer .kg-full-page-img img"

BLOB_INIT_SCRIPT = """
(() => {
  const origCreateObjectURL = URL.createObjectURL.bind(URL);
  URL.createObjectURL = function(blob) {
    const type = blob.type || 'application/octet-stream';
    const url = origCreateObjectURL(blob);
    (async () => {
      const buf = await blob.arrayBuffer();
      let binary = '';
      const bytes = new Uint8Array(buf);
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      const base64 = btoa(binary);
      if (window.captureBlob) {
        window.captureBlob(url, { type, base64 });
      }
    })();
    return url;
  };
})();
"""


def parse_query_params(url_str: str) -> dict[str, str]:
    """Extract flat string dictionary of URL query parameters."""
    parsed = urlparse(url_str)
    raw = parse_qs(parsed.query)
    return {k: v[0] for k, v in raw.items() if v}


def get_toc_items(
    raw_item: AmazonRenderTocItem,
    location_map: AmazonRenderLocationMap | None,
    depth: int = 0,
) -> list[TocItem]:
    """Recursively convert raw render TOC items with coordinate mappings."""
    position_id = raw_item.toc_position_id
    coords: dict[str, int] = {}
    if location_map and location_map.navigation_unit:
        units = location_map.navigation_unit
        for idx, unit in enumerate(units):
            next_start = units[idx + 1].start_position if idx + 1 < len(units) else float("inf")
            if unit.start_position <= position_id < next_start:
                coords = parse_page_label(unit.label)
                break

    item = TocItem(
        label=raw_item.label,
        position_id=position_id,
        page=coords.get("page"),
        roman_page=coords.get("roman_page"),
        depth=depth,
    )
    items = [item]
    if raw_item.entries:
        for child in raw_item.entries:
            items.extend(get_toc_items(child, location_map, depth=depth + 1))
    return items


async def extract_kindle_book(
    asin: str | None = None,
    amazon_email: str | None = None,
    amazon_password: str | None = None,
    out_dir: Path | None = None,
    headless: bool = False,
    context: BrowserContext | None = None,
    max_screens: int | None = None,
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    pacing: PacingConfig | None = None,
    manual_login: bool | None = None,
    login_timeout_seconds: float | None = None,
    resume: bool | None = None,
) -> BookMetadata:
    """Run full browser extraction lifecycle for a Kindle book using Camoufox.

    Non-obvious behavior and design decisions:
    1. Anti-Detect Automation: Uses Camoufox (anti-detect Firefox engine) with randomized
       fingerprinting to bypass Amazon's bot detection on the Cloud Reader interface.
    2. Session & Cookie Persistence: Saves and restores browser storage state to
       ``kindle-session-{domain}-{email_hash}.json`` to preserve login sessions and avoid
       repetitive 2FA verification across extraction runs.
    3. Non-Blocking 2FA Prompting: When Amazon triggers two-factor verification, prompts the
       user via ``asyncio.to_thread(input, ...)`` to keep the asyncio event loop responsive
       and prevent browser websocket heartbeat timeouts while awaiting manual user input.
    4. Network Response Sniffing: Intercepts Cloud Reader internal API responses for book
       manifests, location maps, and table of contents, capturing structural metadata
       that is not directly exposed in the DOM.
    5. Stable Canvas Verification: Before capturing a screenshot, ``stable_state()`` verifies
       that CSS transforms are stationary and that canvas image bytes have converged to avoid
       capturing halfway-rendered or transitioning pages.
    6. Boundary & Loop Detection: Navigation buttons on the Cloud Reader do not always disable
       cleanly at the end of a book. Uses boundary inspection and screen hash comparisons
       to detect when navigation ceases to advance, terminating extraction cleanly.
    7. Human-Like Pacing: Every visible browser action (navigation, clicks, key presses, form fills)
       goes through ``HumanPacer.pause()``, which adds random per-action jitter and enforces a
       rolling-window action budget followed by multi-minute breaks. Polling sleeps that only read
       the DOM are intentionally not paced. Configure via ``pacing`` or ``EXTRACT_*`` env vars.
    8. Manual Login: When ``manual_login`` is true (the default when no credentials are provided),
       the extractor never touches the sign-in form. It waits for the user to complete sign-in and
       2FA in the visible browser window, polling until the reader host is reached, bounded by
       ``login_timeout_seconds`` (``EXTRACT_LOGIN_TIMEOUT``, default 600s). Because the browser
       profile is persistent, later runs with the same output directory reuse the session.
    9. Resumable Extraction: When ``metadata.json`` with captured pages exists and ``resume`` is
       enabled (the default), bypasses rewinding to page 1. Instead, it reuses existing pages,
       jumps to the last captured coordinate, steps forward, and continues appending captures.
       If the existing extraction is already marked complete, it returns immediately without opening
       the browser.
    """
    sleeper = sleep_fn if sleep_fn is not None else asyncio.sleep
    pacing_config = pacing if pacing is not None else PacingConfig.from_env()
    active_asin = asin or get_env("ASIN")
    if not active_asin:
        raise ValueError("ASIN is required")

    active_email = amazon_email or get_env("AMAZON_EMAIL")
    active_password = amazon_password or get_env("AMAZON_PASSWORD")
    use_manual_login = manual_login if manual_login is not None else not (active_email and active_password)
    if not use_manual_login and (not active_email or not active_password):
        raise ValueError("AMAZON_EMAIL and AMAZON_PASSWORD are required unless manual login is enabled")
    if use_manual_login and headless and context is None:
        raise ValueError("Manual login requires a visible browser; disable headless mode")
    login_timeout = login_timeout_seconds if login_timeout_seconds is not None else float(get_env("EXTRACT_LOGIN_TIMEOUT") or 600)

    asin_lower = active_asin.lower()
    raw_domain = get_env("AMAZON_HOST") or get_env("AMAZON_DOMAIN") or "amazon.com"
    domain = reader_domain(raw_domain)
    reader_host = f"read.{domain}"
    book_reader_url = f"https://{reader_host}/?asin={active_asin}"

    base_dir = out_dir or (Path("out") / active_asin)
    user_data_dir = base_dir / "data" / domain
    page_screenshots_dir = base_dir / "pages"
    metadata_path = base_dir / "metadata.json"
    user_data_dir.mkdir(parents=True, exist_ok=True)
    page_screenshots_dir.mkdir(parents=True, exist_ok=True)

    max_screen_count = max_screens or int(get_env("EXTRACT_MAX_SCREENS") or 20_000)
    if max_screen_count <= 0:
        raise ValueError("EXTRACT_MAX_SCREENS must be a positive integer")

    should_resume = (
        resume
        if resume is not None
        else (get_env("EXTRACT_RESUME") or "1").strip().lower() not in ("false", "0", "off")
    )
    existing_meta_data = try_read_json_file(metadata_path) if should_resume else None
    existing_meta: BookMetadata | None = None
    if isinstance(existing_meta_data, dict):
        try:
            existing_meta = BookMetadata.model_validate(existing_meta_data)
        except Exception:
            existing_meta = None

    if should_resume and existing_meta and existing_meta.extraction and existing_meta.extraction.status == "complete":
        log.info("extraction already complete; returning existing metadata", asin=active_asin)
        return existing_meta

    is_resuming = bool(
        should_resume
        and existing_meta
        and existing_meta.pages
        and all(Path(p.screenshot).is_file() for p in existing_meta.pages)
    )

    if is_resuming and existing_meta:
        result = existing_meta
        start_index = len(result.pages)
        beginning = True
        log.info(
            "resuming extraction from previous checkpoint",
            resumed_from_index=start_index - 1,
            next_index=start_index,
        )
    else:
        result = BookMetadata(
            schema_version=2,
            meta=AmazonBookMeta(asin=active_asin, title=active_asin),
            extraction=ExtractionStatus(status="partial", reason="Extraction has not finished"),
            toc=[],
            pages=[],
            nav=Nav(),
        )
        start_index = 0
        beginning = False

    should_close_context = False
    active_context: BrowserContext
    if context is not None:
        active_context = context
    else:
        browser_or_ctx = await AsyncCamoufox(
            persistent_context=True,
            user_data_dir=str(user_data_dir),
            headless=headless,
            firefox_user_prefs={
                "network.trr.mode": 3,
                "network.trr.uri": "https://cloudflare-dns.com/dns-query",
            },
        ).__aenter__()
        if isinstance(browser_or_ctx, BrowserContext):
            active_context = browser_or_ctx
        else:
            active_context = (
                browser_or_ctx.contexts[0]
                if browser_or_ctx.contexts
                else await browser_or_ctx.new_context()
            )
        should_close_context = True

    try:
        pages = active_context.pages
        page: Page = pages[0] if pages else await active_context.new_page()

        captured_blobs: dict[str, dict[str, str]] = {}

        async def on_capture_blob(source: object, url: str, payload: dict[str, str]) -> None:
            blob_type = payload.get("type", "")
            if blob_type and blob_type != "application/octet-stream" and not blob_type.startswith("image/"):
                return
            captured_blobs[url] = payload
            while len(captured_blobs) > 64:
                oldest = next(iter(captured_blobs.keys()))
                del captured_blobs[oldest]

        try:
            await page.expose_binding("captureBlob", on_capture_blob)
        except Exception:
            pass

        try:
            await active_context.add_init_script(BLOB_INIT_SCRIPT)
        except Exception:
            pass

        async def handle_route(route: Route) -> None:
            req = route.request
            url_str = req.url
            for regex in URL_REGEX_BLACKLIST:
                if regex.search(url_str):
                    await route.abort()
                    return

            parsed = urlparse(url_str)
            if parsed.hostname == reader_host and parsed.path == "/renderer/render":
                params = parse_qs(parsed.query)
                params["locationMap"] = ["true"]
                new_query = urlencode(params, doseq=True)
                new_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
                await route.continue_(url=new_url)
                return

            await route.continue_()

        await page.route("**/*", handle_route)

        pending_tasks: set[asyncio.Task[None]] = set()
        network_meta: dict[str, object] = {}
        render_meta: dict[str, object] = {}

        async def handle_response(response: Response) -> None:
            if response.status != 200:
                return

            parsed = urlparse(response.url)
            if parsed.path.endswith("YJmetadata.jsonp"):
                try:
                    body = await response.text()
                    parsed_jsonp = parse_jsonp_response(body)
                    if not parsed_jsonp:
                        return
                    resp_asin = str(parsed_jsonp.get("asin", "")).lower()
                    if resp_asin != asin_lower:
                        return

                    parsed_jsonp.pop("cpr", None)
                    authors_list = parsed_jsonp.get("authorsList")
                    if isinstance(authors_list, list):
                        norm_authors: list[str] = []
                        for author in authors_list:
                            norm_authors.extend(normalize_authors([str(author)]))
                        parsed_jsonp["authorsList"] = norm_authors

                    if parsed_jsonp.get("authorList") is None and parsed_jsonp.get("authorsList") is not None:
                        parsed_jsonp["authorList"] = parsed_jsonp["authorsList"]

                    network_meta.update(parsed_jsonp)
                except Exception:
                    log.warning("unable to process reader jsonp metadata")

            elif parsed.hostname == reader_host:
                if parsed.path == "/service/mobile/reader/startReading":
                    try:
                        body_json = await response.json()
                        if isinstance(body_json, dict):
                            body_asin = str(
                                body_json.get("asin")
                                or body_json.get("ASIN")
                                or body_json.get("requestedAsin")
                                or body_json.get("deliveredAsin")
                                or ""
                            ).lower()
                            if body_asin == asin_lower:
                                body_json.pop("karamelToken", None)
                                body_json.pop("metadataUrl", None)
                                body_json.pop("YJFormatVersion", None)
                                result.info = body_json
                    except Exception:
                        log.warning("unable to process startReading response")

                elif parsed.path == "/renderer/render":
                    try:
                        params = parse_query_params(response.url)
                        render_asin = next(
                            (v for k, v in params.items() if k.lower() == "asin"),
                            None,
                        )
                        if render_asin and render_asin.lower() == asin_lower:
                            render_root = user_data_dir / "render"
                            render_root.mkdir(parents=True, exist_ok=True)
                            temp_render_dir = Path(tempfile.mkdtemp(dir=render_root, prefix="response-"))
                            tar_bytes = await response.body()
                            extract_tar_bytes(tar_bytes, temp_render_dir)

                            loc_map_data = try_read_json_file(temp_render_dir / "location_map.json")
                            if isinstance(loc_map_data, dict):
                                loc_map = AmazonRenderLocationMap.model_validate(loc_map_data)
                                if loc_map.navigation_unit:
                                    for unit in loc_map.navigation_unit:
                                        coords = parse_page_label(unit.label)
                                        unit.page = coords.get("page")
                                        unit.roman_page = coords.get("roman_page")
                                result.location_map = loc_map

                            meta_data = try_read_json_file(temp_render_dir / "metadata.json")
                            if isinstance(meta_data, dict):
                                render_meta.update(meta_data)
                                first_pos = meta_data.get("firstPositionId")
                                last_pos = meta_data.get("lastPositionId")
                                if isinstance(first_pos, int):
                                    result.nav.start_position = first_pos
                                if isinstance(last_pos, int):
                                    result.nav.end_position = last_pos

                            toc_data = try_read_json_file(temp_render_dir / "toc.json")
                            if isinstance(toc_data, list) and not result.toc:
                                toc_items: list[TocItem] = []
                                for raw_toc_item in toc_data:
                                    if isinstance(raw_toc_item, dict):
                                        parsed_item = AmazonRenderTocItem.model_validate(raw_toc_item)
                                        toc_items.extend(get_toc_items(parsed_item, result.location_map))
                                result.toc = toc_items
                    except Exception:
                        log.warning("unable to process reader render response")

        def on_response(res: Response) -> None:
            task = asyncio.create_task(handle_response(res))
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

        page.on("response", on_response)

        async def drain_responses() -> None:
            if not pending_tasks:
                return
            try:
                async with asyncio.timeout(15.0):
                    while pending_tasks:
                        await asyncio.gather(*list(pending_tasks), return_exceptions=True)
            except TimeoutError:
                log.warning("reader metadata processing timed out; optional metadata may be incomplete")

        async def get_page_nav() -> PageNav | None:
            title_el = page.locator("ion-footer ion-title").first
            try:
                text = await title_el.text_content()
                return parse_page_nav(text)
            except Exception:
                return None

        async def stable_state() -> tuple[str, PageNav | None, str]:
            prev_key = ""
            stable_count = 0
            for _ in range(100):
                if cancelled and not restoring:
                    raise RuntimeError("Extraction cancelled")
                img_el = page.locator(KR_RENDERER_IMAGE_SELECTOR)
                src = await img_el.get_attribute("src") or ""
                nav = await get_page_nav()
                key = json.dumps({"src": src, "nav": nav.model_dump() if nav else None}, sort_keys=True)
                loaded = False
                try:
                    loaded = bool(await img_el.evaluate("(el) => el.complete && el.naturalWidth > 0"))
                except Exception:
                    pass
                if src and loaded and key == prev_key:
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= 3:
                    return src, nav, key

                prev_key = key
                await sleeper(0.2)
            raise RuntimeError("Reader did not produce a stable rendered page")

        async def check_boundary(direction: Literal["left", "right"], after_failed_navigation: bool = False) -> bool:
            async def read_snapshot() -> BoundarySnapshot:
                _, _, current_key = await stable_state()
                control = page.locator(f".kr-chevron-container-{direction}").first
                opposite_dir = "right" if direction == "left" else "left"
                opposite = page.locator(f".kr-chevron-container-{opposite_dir}").first

                visible = bool(await control.is_visible())
                disabled = False
                if visible:
                    try:
                        disabled = bool(
                            await control.evaluate(
                                """(el) => el.getAttribute('aria-disabled') === 'true' ||
                                           el.hasAttribute('disabled') ||
                                           !!el.querySelector('[disabled], [aria-disabled="true"]')"""
                            )
                        )
                    except Exception:
                        pass

                opposite_enabled = False
                if await opposite.is_visible():
                    try:
                        is_en = await opposite.is_enabled()
                        aria_dis = await opposite.get_attribute("aria-disabled")
                        opposite_enabled = is_en and aria_dis != "true"
                    except Exception:
                        pass

                return BoundarySnapshot(
                    key=current_key,
                    loaded=True,
                    visible=visible,
                    disabled=disabled,
                    opposite_enabled=opposite_enabled,
                )

            async def pause() -> None:
                await sleeper(0.2)

            return await inspect_boundary(read_snapshot, pause, after_failed_navigation)

        async def dismiss_alerts() -> None:
            try:
                alert_btn = page.locator("ion-alert button, ion-alert ion-button").filter(
                    has_text=re.compile(r"^(No|Cancel|Dismiss)$", re.IGNORECASE)
                ).first
                if await alert_btn.is_visible():
                    await alert_btn.click()
                    await page.locator("ion-alert").wait_for(state="hidden", timeout=3000)
            except Exception:
                pass

        async def capture_rendered_screen() -> bytes:
            await dismiss_alerts()
            try:
                return await page.locator(KR_RENDERER_IMAGE_SELECTOR).screenshot(
                    type="png", scale="css", animations="disabled", timeout=2000
                )
            except Exception:
                box = await page.locator(KR_RENDERER_IMAGE_SELECTOR).bounding_box()
                if box:
                    return await page.screenshot(type="png", clip=box, animations="disabled")
                return await page.screenshot(type="png", animations="disabled")

        async def advance(direction: Literal["left", "right"], old_key: str) -> bool:
            key_name = "ArrowLeft" if direction == "left" else "ArrowRight"
            for attempt in range(4):
                if cancelled and not restoring:
                    break
                await dismiss_alerts()
                if await check_boundary(direction):
                    return False

                chevron = page.locator(f".kr-chevron-container-{direction}").first
                await pacer.pause(f"turn_{direction}")
                if attempt == 0:
                    try:
                        await chevron.click(timeout=3000)
                    except Exception:
                        await dismiss_alerts()
                        await page.keyboard.press(key_name)
                else:
                    await dismiss_alerts()
                    await page.keyboard.press(key_name)
                    try:
                        await chevron.click(timeout=1000)
                    except Exception:
                        pass

                for _ in range(25):
                    if cancelled and not restoring:
                        break
                    await sleeper(0.2)
                    try:
                        _, _, cur_key = await stable_state()
                        if cur_key != old_key:
                            return True
                    except Exception:
                        pass
            return False

        async def go_to_position(target_nav: PageNav) -> None:
            await dismiss_alerts()
            try:
                await page.locator("ion-backdrop, ion-popover").first.wait_for(state="hidden", timeout=3000)
            except Exception:
                pass

            menu_btn = page.locator('ion-button[aria-label="Reader menu"], button[aria-label="Reader menu"]').first
            item = page.locator('ion-item[role="listitem"], ion-item').filter(has_text=re.compile(r"Go to (Page|Location)", re.IGNORECASE)).first

            for _ in range(3):
                if cancelled and not restoring:
                    break
                await pacer.pause("open_reader_menu")
                try:
                    await menu_btn.click(timeout=3000)
                except Exception:
                    try:
                        renderer = page.locator("#kr-renderer")
                        if await renderer.is_visible():
                            box = await renderer.bounding_box()
                            if box:
                                await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + 50)
                            else:
                                await page.mouse.click(400, 50)
                    except Exception:
                        pass

                try:
                    await item.wait_for(state="visible", timeout=3000)
                    break
                except Exception:
                    await sleeper(0.5)

            await pacer.pause("open_go_to")
            try:
                await item.click(timeout=5000)
            except Exception:
                await item.click(force=True)

            modal = page.locator("ion-modal:not(.overlay-hidden), ion-modal.go-to-modal, ion-modal.show-modal").first
            try:
                await modal.wait_for(state="visible", timeout=5000)
            except Exception:
                modal = page.locator("ion-modal").first

            field = page.locator("ion-modal:not(.overlay-hidden) input, ion-modal input").first
            val = target_nav.location if target_nav.location is not None else (target_nav.page or target_nav.label)
            if val is None:
                raise ValueError("Original navigation position is unavailable")

            str_val = str(val)
            await pacer.pause("fill_go_to")
            try:
                await field.fill(str_val)
            except Exception:
                # In real browsers, ion-modal input is input[type=number] which rejects non-digit Roman labels (e.g. 'v').
                # Fall back to page 1 for front matter.
                if not str_val.isdigit():
                    await field.fill("1")
                else:
                    raise
            go_btn = page.locator('[item-i-d="go-to-modal-go-button"], ion-modal ion-button[item-i-d="go-to-modal-go-button"], ion-modal button').filter(has_text=re.compile(r"^Go$", re.IGNORECASE)).first
            await pacer.pause("submit_go_to")
            try:
                await go_btn.click(timeout=5000)
            except Exception:
                await go_btn.click(force=True)

            try:
                await modal.wait_for(state="hidden", timeout=10_000)
            except Exception:
                await page.keyboard.press("Escape")
            await sleeper(0.5)

        async def update_settings() -> None:
            try:
                settings_btn = page.locator('ion-button[aria-label="Reader settings"], button[aria-label="Reader settings"]').first
                await settings_btn.wait_for(timeout=30_000)
                await pacer.pause("open_settings")
                await settings_btn.click()
                await sleeper(0.5)

                ember_btn = page.locator("#AmazonEmber")
                await pacer.pause("select_font")
                await ember_btn.click()
                await sleeper(0.2)

                col_btn = page.locator('[role="radiogroup"][aria-label$=" columns"]').filter(has_text="Single Column")
                await pacer.pause("select_columns")
                await col_btn.click()
                await sleeper(0.2)

                await pacer.pause("close_settings")
                await settings_btn.click()
                await sleeper(0.5)
            except Exception:
                pass

        async def resolve_metadata(final_pass: bool = False) -> None:
            await drain_responses()
            prod_title: str | None = None
            prod_authors: list[str] = []
            prod_cover: str | None = None

            if not final_pass and (not network_meta.get("title") or not network_meta.get("authorList")):
                prod_page = await active_context.new_page()
                try:
                    await pacer.pause("open_product_page")
                    await prod_page.goto(f"https://www.{domain}/dp/{active_asin}", timeout=15_000)
                    title_text = await prod_page.locator("#productTitle").text_content(timeout=3000)
                    if title_text:
                        prod_title = title_text.strip()
                    author_els = await prod_page.locator(".author .a-link-normal").all_text_contents()
                    prod_authors = [a.strip() for a in author_els if a.strip()]
                    cover_el = prod_page.locator("#imgBlkFront, #ebooksImgBlkFront").first
                    prod_cover = await cover_el.get_attribute("src", timeout=3000)
                except Exception:
                    pass
                finally:
                    await prod_page.close()

            title_val = (
                get_env("BOOK_TITLE")
                or str(network_meta.get("title") or "")
                or str(render_meta.get("bookTitle") or "")
                or prod_title
                or active_asin
            )

            env_authors = get_env("BOOK_AUTHOR")
            author_list_raw = network_meta.get("authorList")
            render_authors_raw = render_meta.get("authors")
            if env_authors:
                authors_val = [a.strip() for a in re.split(r"[,;]", env_authors) if a.strip()]
            elif isinstance(author_list_raw, list):
                authors_val = [str(a) for a in author_list_raw]
            elif isinstance(render_authors_raw, list):
                authors_val = [str(a) for a in render_authors_raw]
            elif prod_authors:
                authors_val = prod_authors
            else:
                authors_val = ["Unknown author"]

            result.meta.title = title_val
            result.meta.author_list = authors_val
            cover_url = str(network_meta.get("cover") or prod_cover or "")
            if cover_url:
                result.meta.cover = cover_url

            if result.location_map and result.location_map.navigation_unit:
                pages_found = [u.page for u in result.location_map.navigation_unit if u.page is not None]
                if pages_found:
                    result.nav.total_num_pages = max(pages_found)
                    result.nav.total_num_content_pages = result.nav.total_num_pages
                    result.nav.end_content_page = result.nav.total_num_pages

            if result.meta.cover and not final_pass:
                try:
                    cover_res = await active_context.request.get(result.meta.cover, timeout=15_000)
                    if cover_res.ok:
                        cover_bytes = await cover_res.body()
                        if len(cover_bytes) <= 20_000_000:
                            cover_path = base_dir / "cover.png"
                            with Image.open(io.BytesIO(cover_bytes)) as c_img:
                                c_img.save(cover_path, format="PNG")
                            result.meta.cover_path = str(cover_path)
                except Exception:
                    pass

        cancelled = False
        restoring = False

        def on_signal() -> None:
            nonlocal cancelled
            cancelled = True

        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, on_signal)
            loop.add_signal_handler(signal.SIGTERM, on_signal)
        except (NotImplementedError, RuntimeError):
            pass

        initial_page_nav: PageNav | None = None
        initial_image: bytes | None = None

        # once cancelled, breaks are cut short (even during restoration) so the user isn't stuck waiting
        pacer = HumanPacer(pacing_config, sleep_fn=sleeper, should_abort=lambda: cancelled)

        def on_login_page() -> bool:
            parsed_url = urlparse(page.url)
            return parsed_url.hostname != reader_host or parsed_url.path.startswith("/ap/")

        async def wait_for_manual_login() -> None:
            log.warning(
                "manual login required: sign in (including 2FA) in the browser window; extraction resumes automatically",
                timeout_seconds=int(login_timeout),
            )
            waited = 0.0
            while on_login_page():
                if cancelled:
                    raise RuntimeError("Extraction cancelled")
                if waited >= login_timeout:
                    raise RuntimeError("Timed out waiting for manual login")
                await sleeper(2.0)
                waited += 2.0
            log.info("manual login detected; continuing extraction", waited_seconds=int(waited))

        try:
            await pacer.pause("open_reader")
            await page.goto(book_reader_url, timeout=60_000)

            if use_manual_login and on_login_page():
                await wait_for_manual_login()
                if book_reader_url not in page.url:
                    await pacer.pause("reopen_reader")
                    await page.goto(book_reader_url, timeout=60_000)
            elif "/ap/signin" in page.url:
                assert active_email and active_password
                await pacer.pause("fill_email")
                await page.locator('input[type="email"]').fill(active_email)
                await pacer.pause("submit_email")
                await page.locator("#signInSubmit, #continue").first.click()
                await pacer.pause("fill_password")
                await page.locator('input[type="password"]').fill(active_password)
                await pacer.pause("submit_password")
                await page.locator("#signInSubmit, #continue").first.click()

                if "/kindle-library" not in page.url and "/ap/signin" in page.url:
                    raw_code = await asyncio.to_thread(
                        input,
                        "2-factor auth code (or press Enter if completed in browser): ",
                    )
                    code = raw_code.strip()
                    if code:
                        await pacer.pause("fill_otp")
                        await page.locator('input[type="tel"]').fill(code)
                        await pacer.pause("submit_otp")
                        await page.locator('input[type="submit"][aria-labelledby="cvf-submit-otp-button-announce"]').click()

                if book_reader_url not in page.url:
                    await pacer.pause("reopen_reader")
                    await page.goto(book_reader_url)

            await dismiss_alerts()

            try:
                await page.locator(".top-chrome").evaluate("(el) => { el.style.transition = 'none'; el.style.transform = 'none'; }")
            except Exception:
                pass

            img_locator = page.locator(KR_RENDERER_IMAGE_SELECTOR)
            await img_locator.wait_for(timeout=60_000)
            await update_settings()
            initial_page_nav = await get_page_nav()
            _, _, cur_state_key = await stable_state()
            initial_image = await capture_rendered_screen()
            await resolve_metadata()

            reached_end = False
            if is_resuming:
                last_page = result.pages[-1]
                target_nav = PageNav(
                    page=last_page.page,
                    roman_page=last_page.roman_page,
                    location=last_page.location,
                    label=last_page.label,
                    total=1,
                )
                try:
                    await go_to_position(target_nav)
                except Exception as err:
                    log.error("failed to navigate to target resume position", error=str(err), target=target_nav.model_dump())
                    raise

                _, cur_nav, cur_state_key = await stable_state()
                # Fast forward to beyond the last captured screen
                last_img_path = Path(last_page.screenshot)
                last_img_bytes = last_img_path.read_bytes() if last_img_path.is_file() else None

                # If behind last_page, advance forward until reaching last_page
                if cur_nav and cur_nav.location is not None and last_page.location is not None:
                    while cur_nav and cur_nav.location is not None and cur_nav.location < last_page.location:
                        if not await advance("right", cur_state_key):
                            break
                        _, cur_nav, cur_state_key = await stable_state()

                # If currently displaying the last captured screen, advance one screen to uncaptured territory
                cur_img_bytes = await capture_rendered_screen()
                if (
                    (cur_nav and last_page.location is not None and cur_nav.location is not None and cur_nav.location <= last_page.location)
                    or (last_img_bytes and cur_img_bytes == last_img_bytes)
                    or not last_img_bytes
                ):
                    if not await advance("right", cur_state_key):
                        reached_end = await check_boundary("right", True)
                    else:
                        _, _, cur_state_key = await stable_state()
            else:
                try:
                    await go_to_position(PageNav(page=1, total=1))
                except Exception:
                    await page.keyboard.press("Escape")

                _, _, cur_state_key = await stable_state()
                beginning = False
                for _ in range(max_screen_count):
                    if cancelled:
                        break
                    if await check_boundary("left"):
                        beginning = True
                        break
                    if not await advance("left", cur_state_key):
                        beginning = await check_boundary("left", True)
                        break
                    _, _, cur_state_key = await stable_state()

                if not beginning:
                    result.extraction = ExtractionStatus(
                        status="partial",
                        reason="Unable to confirm the beginning of the book",
                    )

            if not reached_end:
                for index in range(start_index, start_index + max_screen_count):
                    if cancelled:
                        break
                    src, nav, key_before = await stable_state()
                    blob_payload = captured_blobs.get(src)
                    if blob_payload and "base64" in blob_payload:
                        raw_bytes = base64.b64decode(blob_payload["base64"])
                        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                            buf = io.BytesIO()
                            pil_img.save(buf, format="PNG")
                            image_bytes = buf.getvalue()
                    else:
                        image_bytes = await capture_rendered_screen()

                    _, _, key_after = await stable_state()
                    if key_after != key_before:
                        raise RuntimeError("Reader changed during capture")

                    captured_blobs.clear()
                    screenshot_filename = f"{str(index).zfill(6)}.png"
                    screenshot_path = page_screenshots_dir / screenshot_filename
                    screenshot_path.write_bytes(image_bytes)

                    result.pages.append(
                        PageChunk(
                            index=index,
                            page=nav.page if nav else None,
                            roman_page=nav.roman_page if nav else None,
                            location=nav.location if nav else None,
                            label=nav.label if nav else None,
                            screenshot=str(screenshot_path),
                        )
                    )
                    atomic_json(metadata_path, result.model_dump(by_alias=True))

                    if await check_boundary("right"):
                        reached_end = True
                        break
                    if not await advance("right", key_before):
                        reached_end = await check_boundary("right", True)
                        break

            is_full = beginning and reached_end
            status_val: Literal["complete", "partial"] = "complete" if is_full else "partial"
            fail_reason: str | None = None
            if not is_full:
                if cancelled:
                    fail_reason = "Extraction cancelled"
                elif not beginning:
                    fail_reason = "Unable to confirm the beginning of the book"
                else:
                    fail_reason = "Navigation stalled or the capture limit was reached before a confirmed end"

            result.extraction = ExtractionStatus(status=status_val, reason=fail_reason)
        except Exception as err:
            err_msg = str(err)
            safe_messages = {
                "Reader did not produce a stable rendered page",
                "Reader changed during capture",
                "EXTRACT_MAX_SCREENS must be a positive integer",
                "Extraction cancelled",
                "Timed out waiting for manual login",
            }
            fail_reason = "Extraction cancelled" if cancelled else (err_msg if err_msg in safe_messages else "Reader interaction or image capture failed")
            result.extraction = ExtractionStatus(status="partial", reason=fail_reason)
            log.warning("reader interaction or image capture failed", error=fail_reason, raw_error=err_msg)
        finally:
            restoring = True
            try:
                if initial_page_nav:
                    await go_to_position(initial_page_nav)
                    result.extraction = result.extraction or ExtractionStatus()
                    result.extraction.restoration = "failed"
                    for _ in range(100):
                        _, nav, restored_key = await stable_state()
                        if (
                            (nav.page if nav else None) != initial_page_nav.page
                            or (nav.roman_page if nav else None) != initial_page_nav.roman_page
                            or (nav.location if nav else None) != initial_page_nav.location
                        ):
                            break
                        restored_img = await capture_rendered_screen()
                        if initial_image == restored_img:
                            result.extraction.restoration = "succeeded"
                            break
                        if not await advance("right", restored_key):
                            break
                else:
                    if result.extraction:
                        result.extraction.restoration = "unavailable"
            except Exception:
                if result.extraction:
                    result.extraction.restoration = "failed"

            await resolve_metadata(final_pass=True)
            atomic_json(metadata_path, result.model_dump(by_alias=True))
            if pacing_config.enabled:
                log.info(
                    "pacing summary",
                    breaks_taken=pacer.breaks_taken,
                    paused_seconds=round(pacer.total_paused_seconds, 1),
                )

            try:
                loop.remove_signal_handler(signal.SIGINT)
                loop.remove_signal_handler(signal.SIGTERM)
            except (NotImplementedError, RuntimeError):
                pass
    finally:
        if should_close_context and active_context is not None:
            await active_context.close()

    return result
