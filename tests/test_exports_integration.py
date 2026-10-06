"""Integration tests for Markdown and PDF export workflows."""

import json
import shutil
import subprocess
from pathlib import Path
from PIL import Image
import pytest

from python_kindle_ai_export.export_markdown import export_markdown
from python_kindle_ai_export.export_pdf import export_pdf

has_pdfinfo = shutil.which("pdfinfo") is not None
has_pdftotext = shutil.which("pdftotext") is not None


def get_pdf_page_count(pdf_path: Path) -> int:
    res = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True)
    for line in res.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":")[1].strip())
    raise ValueError("Page count not found")


def get_pdf_text(pdf_path: Path) -> str:
    res = subprocess.run(["pdftotext", str(pdf_path), "-"], capture_output=True, text=True, check=True)
    return res.stdout


@pytest.fixture
def export_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    asin = "test-export-asin"
    monkeypatch.setenv("ASIN", asin)
    monkeypatch.setenv("OUTPUT_BASENAME", "fixture")

    out_dir = tmp_path / "out" / asin
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = ["FIRST CAPTURE", "MIDDLE CAPTURE", "FINAL CAPTURE"]
    content: list[dict[str, object]] = []

    for i in range(3):
        screenshot = out_dir / f"{i}.png"
        img = Image.new("RGB", (100, 150), color="#abcdef" if i % 2 else "#ffffff")
        img.save(screenshot, format="PNG")
        content.append(
            {
                "index": i,
                "page": i + 1,
                "screenshot": str(screenshot),
                "text": texts[i],
            }
        )

    metadata = {
        "meta": {
            "asin": asin,
            "title": "Fixture Book",
            "authorList": ["Test Author"],
        },
        "pages": [
            {k: v for k, v in c.items() if k != "text"}
            for c in content
        ],
        "toc": [
            {"label": "Repeated title", "page": 1, "positionId": 1, "depth": 0},
            {"label": "Repeated title", "page": 3, "positionId": 3, "depth": 0},
        ],
    }

    (out_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (out_dir / "content.json").write_text(json.dumps(content), encoding="utf-8")

    return out_dir, content


def test_writes_every_markdown_capture_once_and_unique_links(export_fixture) -> None:
    out_dir, content = export_fixture
    md_path = export_markdown(out_dir=out_dir)

    assert md_path.is_file()
    md_content = md_path.read_text(encoding="utf-8")

    for chunk in content:
        assert str(chunk["text"]) in md_content
        # Exactly one occurrence of chunk text
        assert md_content.count(str(chunk["text"])) == 1

    assert "](#chapter-1)" in md_content
    assert "](#chapter-2)" in md_content
    assert 'id="chapter-1"' in md_content
    assert not any(p.name.endswith(".tmp") for p in out_dir.iterdir())


@pytest.mark.skipif(not (has_pdfinfo and has_pdftotext), reason="poppler utilities required")
def test_generates_text_pdf_with_chapter_content(export_fixture) -> None:
    out_dir, content = export_fixture
    pdf_path = export_pdf(out_dir=out_dir, mode="text")

    assert get_pdf_page_count(pdf_path) == 3
    extracted_text = get_pdf_text(pdf_path)
    for chunk in content:
        assert str(chunk["text"]) in extracted_text


@pytest.mark.skipif(not has_pdfinfo, reason="poppler utilities required")
def test_exports_images_pdf_without_transcription(export_fixture) -> None:
    out_dir, _ = export_fixture
    (out_dir / "content.json").unlink()
    pdf_path = export_pdf(out_dir=out_dir, mode="images")
    assert get_pdf_page_count(pdf_path) == 3


@pytest.mark.skipif(not has_pdfinfo, reason="poppler utilities required")
def test_auto_mode_falls_back_to_images_for_incomplete_text(export_fixture, capsys: pytest.CaptureFixture) -> None:
    out_dir, content = export_fixture
    # Incomplete content (only 1 of 3 pages)
    (out_dir / "content.json").write_text(json.dumps(content[:1]), encoding="utf-8")

    pdf_path = export_pdf(out_dir=out_dir, mode="auto")
    assert get_pdf_page_count(pdf_path) == 3
    captured = capsys.readouterr()
    assert "incomplete" in captured.err


def test_text_mode_rejects_incomplete_transcription_and_preserves_existing(export_fixture) -> None:
    out_dir, content = export_fixture
    (out_dir / "content.json").write_text(json.dumps(content[:1]), encoding="utf-8")
    fixture_pdf = out_dir / "fixture.pdf"
    fixture_pdf.write_text("existing export", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete"):
        export_pdf(out_dir=out_dir, mode="text")

    assert fixture_pdf.read_text(encoding="utf-8") == "existing export"


@pytest.mark.asyncio
async def test_export_audio_openai_and_id3(export_fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock, MagicMock, patch
    from python_kindle_ai_export.export_audio import export_audio

    out_dir, _ = export_fixture
    fake_audio = b"ID3\x03\x00\x00\x00\x00\x00\x00"

    with patch("python_kindle_ai_export.export_audio.generate_openai_speech", new=AsyncMock(return_value=fake_audio)), \
         patch("subprocess.run") as mock_subproc, \
         patch("python_kindle_ai_export.export_audio.ID3") as mock_id3:
        mock_subproc.return_value = MagicMock(returncode=0)
        mock_id3.return_value = MagicMock()

        audio_path = await export_audio(out_dir=out_dir, engine="openai", preview=False)
        assert audio_path.name.endswith(".mp3")
        assert mock_subproc.called
        assert mock_id3.called


@pytest.mark.asyncio
async def test_export_audio_unrealspeech_preview(export_fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock, MagicMock, patch
    from python_kindle_ai_export.export_audio import export_audio

    out_dir, _ = export_fixture
    fake_audio = b"ID3\x03\x00\x00\x00\x00\x00\x00"

    with patch("python_kindle_ai_export.export_audio.generate_unrealspeech", new=AsyncMock(return_value=fake_audio)), \
         patch("subprocess.run") as mock_subproc, \
         patch("python_kindle_ai_export.export_audio.ID3") as mock_id3:
        mock_subproc.return_value = MagicMock(returncode=0)
        mock_id3.return_value = MagicMock()

        audio_path = await export_audio(out_dir=out_dir, engine="unrealspeech", preview=True)
        assert "- preview.mp3" in audio_path.name
        assert mock_subproc.called
