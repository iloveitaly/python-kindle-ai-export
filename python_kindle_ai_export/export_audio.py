"""Export transcribed book content to an audiobook or audio preview using TTS and ffmpeg."""

import asyncio
import logging
import subprocess
from pathlib import Path
import httpx
from mutagen.id3 import ID3, TIT2, TPE1, WCOM

from .book_chapters import audio_batches, book_chapters, narration_content
from .book_data import (
    assert_complete_extraction,
    assert_complete_transcription,
    read_book_content,
    read_book_metadata,
)
from .book_filename import book_filename
from .log import log
from .utils import canonical_json_digest, get_env


async def generate_openai_speech(
    text: str,
    voice: str = "alloy",
    model: str = "tts-1-hd",
    api_key: str | None = None,
) -> bytes:
    """Generate MP3 audio bytes using OpenAI Speech API."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    response = await client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format="mp3",
    )
    return await response.aread()


async def generate_unrealspeech(
    text: str,
    voice_id: str = "Scarlett",
    api_key: str | None = None,
) -> bytes:
    """Generate MP3 audio bytes using UnrealSpeech API."""
    key = api_key or get_env("UNREAL_SPEECH_API_KEY")
    headers = {"Authorization": f"Bearer {key}"}
    payload = {"Text": text, "VoiceId": voice_id}
    async with httpx.AsyncClient() as client:
        res = await client.post(
            "https://api.unrealspeech.com/stream",
            json=payload,
            headers=headers,
            timeout=60.0,
        )
        res.raise_for_status()
        return res.content


async def export_audio(
    asin: str | None = None,
    out_dir: Path | None = None,
    preview: bool | None = None,
    force: bool = False,
    engine: str | None = None,
) -> Path:
    """Generate complete audiobook or short preview and tag with ID3 metadata."""
    active_asin = asin or get_env("ASIN")
    if out_dir is not None:
        base_dir = out_dir
    elif active_asin:
        base_dir = Path("out") / active_asin
    else:
        raise ValueError("ASIN is required")

    is_preview = (
        preview
        if preview is not None
        else (get_env("AUDIOBOOK_PREVIEW") == "true")
    )
    is_force = force or (get_env("FORCE") == "true")
    tts_engine = (engine or get_env("TTS_ENGINE") or "openai").lower()
    if tts_engine not in ("openai", "unrealspeech"):
        raise ValueError(f'Invalid TTS engine "{tts_engine}"')

    voice = (
        get_env("OPENAI_TTS_VOICE") or "alloy"
        if tts_engine == "openai"
        else (get_env("UNREAL_SPEECH_VOICE") or "Scarlett")
    )

    metadata = read_book_metadata(base_dir / "metadata.json")
    raw_content = read_book_content(base_dir / "content.json")
    assert_complete_extraction(metadata)
    assert_complete_transcription(base_dir, metadata, raw_content)

    content = narration_content(raw_content, is_preview)
    if not content:
        raise ValueError("no book content found")

    title = metadata.meta.title
    authors = metadata.meta.author_list
    max_chars = 4096 if tts_engine == "openai" else 3000

    config_hash = canonical_json_digest(
        {
            "engine": tts_engine,
            "voice": voice,
            "title": title,
            "authors": authors,
            "chunks": len(content),
        }
    )[:12]

    subfolder = "audio-previews" if is_preview else "audio"
    config_dir_name = f"{tts_engine}-{voice}-{config_hash}"
    tts_out_dir = base_dir / subfolder / config_dir_name
    tts_out_dir.mkdir(parents=True, exist_ok=True)

    batches: list[str] = [
        f"{'Audiobook Preview of ' if is_preview else ''}{title}\n\nBy {', '.join(authors)}"
    ]

    for chapter in book_chapters(content, metadata.toc):
        chapter_text = f"{chapter.label}\n\n" + "\n\n".join(
            chunk.text for chunk in chapter.chunks
        )
        batches.extend(audio_batches(chapter_text, max_chars))

    if is_preview:
        batches.append("End of preview")

    padding = len(str(len(batches)))
    audio_files: list[Path] = []
    semaphore = asyncio.Semaphore(16)

    async def generate_batch(index: int, text: str) -> Path:
        filename = f"{str(index).zfill(padding)}.mp3"
        target_path = tts_out_dir / filename
        if not is_force and target_path.is_file():
            return target_path

        async with semaphore:
            if tts_engine == "openai":
                audio_bytes = await generate_openai_speech(text, voice=voice)
            else:
                audio_bytes = await generate_unrealspeech(text, voice_id=voice)

            target_path.write_bytes(audio_bytes)
            return target_path

    audio_files = await asyncio.gather(
        *(generate_batch(i, text) for i, text in enumerate(batches))
    )

    concat_manifest = tts_out_dir / "files.txt"
    manifest_content = "\n".join(f"file '{f.name}'" for f in audio_files)
    concat_manifest.write_text(manifest_content, encoding="utf-8")

    suffix = " - preview" if is_preview else ""
    output_audio_path = tts_out_dir / f"{book_filename(metadata.meta)}{suffix}.mp3"

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_manifest),
        "-c",
        "copy",
        "-metadata",
        f"title={'Preview of ' if is_preview else ''}{title}",
        "-metadata",
        f"artist={'/'.join(authors)}",
        str(output_audio_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    try:
        audio_id3 = ID3(output_audio_path)
        audio_id3.add(
            TIT2(
                encoding=3,
                text=[f"{'Preview of ' if is_preview else ''}{title}"],
            )
        )
        audio_id3.add(TPE1(encoding=3, text=["/".join(authors)]))
        audio_id3.add(
            WCOM(url=f"https://www.amazon.com/dp/{metadata.meta.asin}")
        )
        audio_id3.save()
    except Exception as err:
        log.warning("failed to add extra ID3 metadata", error=str(err))

    return output_audio_path
