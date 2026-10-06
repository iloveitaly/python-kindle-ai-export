"""Tests for Click CLI commands."""

from click.testing import CliRunner
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import pytest

from python_kindle_ai_export.cli import cli


def test_cli_version() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "extract" in result.output
    assert "transcribe" in result.output
    assert "export-markdown" in result.output
    assert "export-pdf" in result.output
    assert "export-audio" in result.output
    assert "export-all" in result.output


def test_cli_extract_invokes_extractor(tmp_path: Path) -> None:
    runner = CliRunner()
    mock_res = MagicMock()
    mock_res.extraction.status = "complete"
    with patch("python_kindle_ai_export.cli.extract_kindle_book", new=AsyncMock(return_value=mock_res)) as mock_ext:
        result = runner.invoke(
            cli,
            [
                "extract",
                "--asin", "B00TEST",
                "--email", "test@example.com",
                "--password", "secret",
                "--out-dir", str(tmp_path),
                "--headless",
                "--max-screens", "5",
            ],
        )
        assert result.exit_code == 0
        assert mock_ext.called
        assert mock_ext.call_args.kwargs["resume"] is True


def test_cli_extract_respects_no_resume(tmp_path: Path) -> None:
    runner = CliRunner()
    mock_res = MagicMock()
    mock_res.extraction.status = "complete"
    with patch("python_kindle_ai_export.cli.extract_kindle_book", new=AsyncMock(return_value=mock_res)) as mock_ext:
        result = runner.invoke(
            cli,
            [
                "extract",
                "--asin", "B00TEST",
                "--email", "test@example.com",
                "--password", "secret",
                "--out-dir", str(tmp_path),
                "--headless",
                "--no-resume",
            ],
        )
        assert result.exit_code == 0
        assert mock_ext.called
        assert mock_ext.call_args.kwargs["resume"] is False


def test_cli_transcribe_invokes_transcriber(tmp_path: Path) -> None:
    runner = CliRunner()
    mock_meta = MagicMock()
    with patch("python_kindle_ai_export.cli.read_book_metadata", return_value=mock_meta), \
         patch("python_kindle_ai_export.cli.assert_complete_extraction"), \
         patch("python_kindle_ai_export.cli.create_transcription_provider", return_value=MagicMock()), \
         patch("python_kindle_ai_export.cli.transcribe_pages", new=AsyncMock(return_value=MagicMock())) as mock_tx:
        result = runner.invoke(
            cli,
            [
                "transcribe",
                "--asin", "B00TEST",
                "--out-dir", str(tmp_path),
                "--provider", "openai",
                "--model", "gpt-4o",
            ],
        )
        assert result.exit_code == 0
        assert mock_tx.called


def test_cli_export_markdown(tmp_path: Path) -> None:
    runner = CliRunner()
    target = tmp_path / "book.md"
    with patch("python_kindle_ai_export.cli.run_export_markdown", return_value=target) as mock_md:
        result = runner.invoke(
            cli,
            ["export-markdown", "--asin", "B00TEST", "--out-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert mock_md.called


def test_cli_export_pdf(tmp_path: Path) -> None:
    runner = CliRunner()
    target = tmp_path / "book.pdf"
    with patch("python_kindle_ai_export.cli.run_export_pdf", return_value=target) as mock_pdf:
        result = runner.invoke(
            cli,
            ["export-pdf", "--asin", "B00TEST", "--out-dir", str(tmp_path), "--mode", "images"],
        )
        assert result.exit_code == 0
        assert mock_pdf.called


def test_cli_export_audio(tmp_path: Path) -> None:
    runner = CliRunner()
    target = tmp_path / "book.mp3"
    with patch("python_kindle_ai_export.cli.run_export_audio", new=AsyncMock(return_value=target)) as mock_audio:
        result = runner.invoke(
            cli,
            ["export-audio", "--asin", "B00TEST", "--out-dir", str(tmp_path), "--preview"],
        )
        assert result.exit_code == 0
        assert mock_audio.called


def test_cli_export_all(tmp_path: Path) -> None:
    runner = CliRunner()
    with patch("python_kindle_ai_export.cli.run_export_markdown", return_value=tmp_path / "book.md") as mock_md, \
         patch("python_kindle_ai_export.cli.run_export_pdf", return_value=tmp_path / "book.pdf") as mock_pdf:
        result = runner.invoke(
            cli,
            ["export-all", "--asin", "B00TEST", "--out-dir", str(tmp_path), "--mode", "auto"],
        )
        assert result.exit_code == 0
        assert mock_md.called
        assert mock_pdf.called
