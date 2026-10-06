"""General utility and filesystem functions."""

import hashlib
import io
import json
import os
import re
import tarfile
import uuid
from pathlib import Path
from pydantic import JsonValue

ROMAN_NUMERALS: dict[str, int] = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}

JSONP_REGEX = re.compile(r"\((\{[\s\S]*\})\)")


def get_env(name: str) -> str | None:
    """Read an environment variable, prioritizing KINDLE_AI_EXPORT_ prefix."""
    if name.startswith("KINDLE_AI_EXPORT_"):
        raw_name = name[len("KINDLE_AI_EXPORT_") :]
        return (
            os.environ.get(name)
            or os.environ.get(f"KINDLE_EXPORT_{raw_name}")
            or os.environ.get(raw_name)
        )
    return (
        os.environ.get(f"KINDLE_AI_EXPORT_{name}")
        or os.environ.get(f"KINDLE_EXPORT_{name}")
        or os.environ.get(name)
    )


def normalize_authors(raw_authors: list[str]) -> list[str]:
    """Normalize and deduplicate author strings from Kindle metadata."""
    if not raw_authors:
        return []

    raw_author = raw_authors[0]
    author_tokens = [token.strip() for token in raw_author.split(":") if token.strip()]
    unique_tokens = list(dict.fromkeys(author_tokens))

    normalized: list[str] = []
    for entry in unique_tokens:
        parts = [part.strip() for part in entry.split(",") if part.strip()]
        parts.reverse()
        normalized.append(" ".join(parts))
    return normalized


def parse_jsonp_response(body: str) -> dict[str, object] | None:
    """Extract and parse the JSON payload from a JSONP response string."""
    match = JSONP_REGEX.search(body)
    if not match:
        return None

    content = match.group(1).strip()
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        return None
    return None


def deromanize(roman_numeral: str) -> int:
    """Convert a Roman numeral string to an integer."""
    letters = list(roman_numeral.upper())
    num = 0
    while letters:
        current_char = letters.pop(0)
        current_val = ROMAN_NUMERALS.get(current_char, 0)
        if letters and current_val < ROMAN_NUMERALS.get(letters[0], 0):
            num -= current_val
        else:
            num += current_val
    return num


def digest(value: bytes | str) -> str:
    """Compute sha256 hex digest of bytes or string."""
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def canonical_json_str(data: object) -> str:
    """Produce deterministic compact JSON string matching JSON.stringify."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def canonical_json_digest(data: object) -> str:
    """Compute sha256 digest of canonically formatted JSON data."""
    return digest(canonical_json_str(data))


def atomic_write(file_path: Path, data: str | bytes) -> None:
    """Write data to a temporary file on the same filesystem, then replace."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_name(f"{file_path.name}.{uuid.uuid4()}.tmp")
    try:
        if isinstance(data, str):
            temp_path.write_text(data, encoding="utf-8")
        else:
            temp_path.write_bytes(data)
        temp_path.replace(file_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def atomic_json(file_path: Path, data: object, indent: int | None = 2) -> None:
    """Write an object as JSON atomically."""
    serialized = json.dumps(data, indent=indent, ensure_ascii=False)
    atomic_write(file_path, serialized)


def read_json_file(file_path: Path) -> JsonValue:
    """Read and parse a JSON file from disk."""
    return json.loads(file_path.read_text(encoding="utf-8"))


def try_read_json_file(file_path: Path) -> JsonValue | None:
    """Read a JSON file from disk, returning None if missing or invalid."""
    if not file_path.is_file():
        return None
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def extract_tar_bytes(buf: bytes, cwd: Path) -> Path:
    """Decompress a TAR or TGZ byte buffer into a target directory."""
    cwd.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(buf)) as archive:
        for member in archive.getmembers():
            target = (cwd / member.name).resolve()
            if not str(target).startswith(str(cwd.resolve())):
                raise ValueError(f"Path traversal detected in archive: {member.name}")
        archive.extractall(path=cwd)
    return cwd
