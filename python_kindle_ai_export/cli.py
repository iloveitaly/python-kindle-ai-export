"""Command-line interface for Kindle book extraction, transcription, and exports."""

import asyncio
import os
import sys
from pathlib import Path
import click
from dotenv import load_dotenv

from .book_data import assert_complete_extraction, read_book_metadata
from .export_audio import export_audio as run_export_audio
from .export_markdown import export_markdown as run_export_markdown
from .export_pdf import export_pdf as run_export_pdf
from .extractor import extract_kindle_book
from .log import log
from .providers import create_transcription_provider
from .transcription import transcribe_pages
from .utils import get_env
from .version import __version__

load_dotenv()


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Export books from Kindle web reader with AI transcription and document exports."""


@cli.command()
@click.option("--asin", envvar=["KINDLE_AI_EXPORT_ASIN", "ASIN"], help="Amazon ASIN of the Kindle book")
@click.option("--email", envvar=["KINDLE_AI_EXPORT_AMAZON_EMAIL", "AMAZON_EMAIL", "EMAIL"], help="Amazon account email")
@click.option("--password", envvar=["KINDLE_AI_EXPORT_AMAZON_PASSWORD", "AMAZON_PASSWORD", "PASSWORD"], help="Amazon account password")
@click.option("--out-dir", type=click.Path(path_type=Path), envvar=["KINDLE_AI_EXPORT_OUT_DIR", "OUT_DIR"], help="Output directory")
@click.option("--headless/--no-headless", default=False, envvar=["KINDLE_AI_EXPORT_HEADLESS", "HEADLESS"], help="Run browser in headless mode")
@click.option("--max-screens", type=int, default=None, envvar=["KINDLE_AI_EXPORT_MAX_SCREENS", "EXTRACT_MAX_SCREENS"], help="Maximum screen captures")
@click.option(
    "--manual-login/--auto-login",
    default=None,
    envvar=["KINDLE_AI_EXPORT_MANUAL_LOGIN", "EXTRACT_MANUAL_LOGIN"],
    help="Sign in (incl. 2FA) yourself in the browser window. Defaults to manual when no email/password is set.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    envvar=["KINDLE_AI_EXPORT_RESUME", "EXTRACT_RESUME"],
    help="Resume extraction from existing partial progress if present",
)
def extract(
    asin: str | None,
    email: str | None,
    password: str | None,
    out_dir: Path | None,
    headless: bool,
    max_screens: int | None,
    manual_login: bool | None,
    resume: bool,
) -> None:
    """Extract rendered book screens and metadata using camoufox anti-detect browser."""
    if not asin:
        raise click.UsageError("ASIN is required (--asin or KINDLE_AI_EXPORT_ASIN environment variable)")
    use_manual = manual_login if manual_login is not None else not (email and password)
    if not use_manual and (not email or not password):
        raise click.UsageError("AMAZON_EMAIL and AMAZON_PASSWORD are required with --auto-login")
    if use_manual and headless:
        raise click.UsageError("Manual login requires a visible browser; use --no-headless")

    result = asyncio.run(
        extract_kindle_book(
            asin=asin,
            amazon_email=email,
            amazon_password=password,
            out_dir=out_dir,
            headless=headless,
            max_screens=max_screens,
            manual_login=use_manual,
            resume=resume,
        )
    )
    if result.extraction and result.extraction.status != "complete":
        click.secho(f"Extraction partial: {result.extraction.reason}", fg="yellow")
        sys.exit(1)

    click.secho(f"Extraction complete for {asin}", fg="green")


@cli.command()
@click.option("--asin", envvar=["KINDLE_AI_EXPORT_ASIN", "ASIN"], help="Amazon ASIN of the Kindle book")
@click.option("--out-dir", type=click.Path(path_type=Path), envvar=["KINDLE_AI_EXPORT_OUT_DIR", "OUT_DIR"], help="Output directory")
@click.option(
    "--provider",
    default=None,
    envvar=["KINDLE_AI_EXPORT_PROVIDER", "KINDLE_EXPORT_PROVIDER", "AI_PROVIDER", "PROVIDER"],
    help="AI provider (openai, gemini, anthropic, groq, etc.)",
)
@click.option(
    "--model",
    default=None,
    envvar=["KINDLE_AI_EXPORT_MODEL", "KINDLE_EXPORT_MODEL", "MODEL", "AI_MODEL"],
    help="Model identifier (e.g. google:gemini-3.8-flash, openai:gpt-4o)",
)
@click.option(
    "--api-key",
    default=None,
    envvar=["KINDLE_AI_EXPORT_AI_KEY", "KINDLE_EXPORT_AI_KEY", "AI_KEY"],
    help="Provider API key",
)
@click.option("--concurrency", type=int, default=3, envvar=["KINDLE_AI_EXPORT_CONCURRENCY", "TRANSCRIBE_CONCURRENCY"], help="Concurrent request limit")
@click.option("--max-attempts", type=int, default=100, envvar=["KINDLE_AI_EXPORT_MAX_ATTEMPTS", "TRANSCRIBE_MAX_ATTEMPTS"], help="Maximum attempts per capture (defaults to 100 for patient multi-hour retries)")
@click.option("--timeout-ms", type=int, default=120_000, envvar=["KINDLE_AI_EXPORT_TIMEOUT_MS", "TRANSCRIBE_TIMEOUT_MS"], help="Per-request timeout in milliseconds")
def transcribe(
    asin: str | None,
    out_dir: Path | None,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    concurrency: int,
    max_attempts: int,
    timeout_ms: int,
) -> None:
    """Transcribe captured book screens using pydantic-ai vision models."""
    active_asin = asin or get_env("ASIN")
    if out_dir is not None:
        base_dir = out_dir
    elif active_asin:
        base_dir = Path("out") / active_asin
    else:
        raise click.UsageError("ASIN is required")
    metadata = read_book_metadata(base_dir / "metadata.json")
    assert_complete_extraction(metadata)

    prov = create_transcription_provider(provider, model, api_key)

    asyncio.run(
        transcribe_pages(
            pages=metadata.pages,
            out_dir=base_dir,
            provider=prov,
            concurrency=concurrency,
            max_attempts=max_attempts,
            timeout_ms=timeout_ms,
        )
    )
    click.secho(
        f"Transcribed {len(metadata.pages)} pages with {prov.name}/{prov.model}",
        fg="green",
    )


@cli.command("export-markdown")
@click.option("--asin", envvar=["KINDLE_AI_EXPORT_ASIN", "ASIN"], help="Amazon ASIN of the Kindle book")
@click.option("--out-dir", type=click.Path(path_type=Path), envvar=["KINDLE_AI_EXPORT_OUT_DIR", "OUT_DIR"], help="Output directory")
def export_markdown(asin: str | None, out_dir: Path | None) -> None:
    """Export transcribed book to Markdown format."""
    output_path = run_export_markdown(asin=asin, out_dir=out_dir)
    click.secho(f"Generated Markdown: {output_path}", fg="green")


@cli.command("export-pdf")
@click.option("--asin", envvar=["KINDLE_AI_EXPORT_ASIN", "ASIN"], help="Amazon ASIN of the Kindle book")
@click.option("--out-dir", type=click.Path(path_type=Path), envvar=["KINDLE_AI_EXPORT_OUT_DIR", "OUT_DIR"], help="Output directory")
@click.option("--mode", type=click.Choice(["auto", "text", "images"]), default="auto", envvar=["KINDLE_AI_EXPORT_PDF_MODE", "PDF_MODE"], help="PDF generation mode")
def export_pdf(asin: str | None, out_dir: Path | None, mode: str) -> None:
    """Export book to PDF (images, text, or auto)."""
    output_path = run_export_pdf(asin=asin, out_dir=out_dir, mode=mode)
    click.secho(f"Generated PDF: {output_path}", fg="green")


@cli.command("export-audio")
@click.option("--asin", envvar=["KINDLE_AI_EXPORT_ASIN", "ASIN"], help="Amazon ASIN of the Kindle book")
@click.option("--out-dir", type=click.Path(path_type=Path), envvar=["KINDLE_AI_EXPORT_OUT_DIR", "OUT_DIR"], help="Output directory")
@click.option("--preview/--no-preview", default=False, envvar=["KINDLE_AI_EXPORT_AUDIOBOOK_PREVIEW", "AUDIOBOOK_PREVIEW"], help="Export only first capture as preview")
@click.option("--force/--no-force", default=False, envvar=["KINDLE_AI_EXPORT_FORCE", "FORCE"], help="Regenerate all audio batches")
@click.option("--engine", type=click.Choice(["openai", "unrealspeech"]), default="openai", envvar=["KINDLE_AI_EXPORT_TTS_ENGINE", "TTS_ENGINE"], help="TTS engine")
def export_audio(
    asin: str | None,
    out_dir: Path | None,
    preview: bool,
    force: bool,
    engine: str,
) -> None:
    """Export transcribed book to audiobook format via TTS."""
    output_path = asyncio.run(
        run_export_audio(
            asin=asin,
            out_dir=out_dir,
            preview=preview,
            force=force,
            engine=engine,
        )
    )
    click.secho(f"Generated audiobook: {output_path}", fg="green")


@cli.command("export-all")
@click.option("--asin", envvar=["KINDLE_AI_EXPORT_ASIN", "ASIN"], help="Amazon ASIN of the Kindle book")
@click.option("--out-dir", type=click.Path(path_type=Path), envvar=["KINDLE_AI_EXPORT_OUT_DIR", "OUT_DIR"], help="Output directory")
@click.option("--mode", type=click.Choice(["auto", "text", "images"]), default="auto", envvar=["KINDLE_AI_EXPORT_PDF_MODE", "PDF_MODE"], help="PDF generation mode")
def export_all(asin: str | None, out_dir: Path | None, mode: str) -> None:
    """Export book to both Markdown and PDF formats."""
    md_path = run_export_markdown(asin=asin, out_dir=out_dir)
    pdf_path = run_export_pdf(asin=asin, out_dir=out_dir, mode=mode)
    click.secho(f"Generated Markdown: {md_path}", fg="green")
    click.secho(f"Generated PDF: {pdf_path}", fg="green")


def main() -> None:
    """Entry point for the command-line interface."""
    cli()


if __name__ == "__main__":
    main()
