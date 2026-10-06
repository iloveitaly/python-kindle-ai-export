"""Data models for kindle book metadata, extraction, and transcription."""

from typing import Literal
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue
from pydantic.alias_generators import to_camel


class BaseKindleModel(BaseModel):
    """Base model with camelCase serialization and deserialization support."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        serialize_by_alias=True,
    )


class PageChunk(BaseKindleModel):
    """A captured screen; index is the stable ordinal, not a page number."""

    index: int
    page: int | None = None
    roman_page: int | None = None
    location: int | None = None
    position: int | None = None
    end_position: int | None = None
    label: str | None = None
    screenshot: str


class ContentChunk(PageChunk):
    """A transcribed screen with captured coordinates and extracted text."""

    text: str


class PageNav(BaseKindleModel):
    """Parsed reader navigation footer coordinates."""

    page: int | None = None
    roman_page: int | None = None
    location: int | None = None
    label: str | None = None
    total: int = 1


class TocItem(BaseKindleModel):
    """Table of contents item with nesting depth and mapped positions."""

    label: str
    position_id: int
    page: int | None = None
    roman_page: int | None = None
    location: int | None = None
    depth: int = 0


class AmazonBookPositions(BaseKindleModel):
    """Standard position IDs from book metadata."""

    cover: int | None = None
    srl: int = 0
    toc: int = 0


class AmazonBookMeta(BaseKindleModel):
    """Amazon book metadata from reader network responses."""

    acr: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ACR", "acr"),
        serialization_alias="ACR",
    )
    asin: str
    author_list: list[str] = Field(default_factory=list)
    book_size: str | None = None
    book_type: str | None = None
    cover: str | None = None
    cover_path: str | None = None
    language: str | None = None
    positions: AmazonBookPositions | dict[str, int] | None = None
    publisher: str | None = None
    ref_em_id: str | None = None
    release_date: str | None = None
    sample: bool | None = None
    title: str
    version: str | None = None
    start_position: int | None = None
    end_position: int | None = None


class AmazonRenderNavigationUnit(BaseKindleModel):
    """Reader navigation unit from render response location map."""

    start_position: int
    page: int | None = None
    roman_page: int | None = None
    label: str


class AmazonRenderLocationMap(BaseKindleModel):
    """Reader location map from render archive."""

    locations: list[int] = Field(default_factory=list)
    navigation_unit: list[AmazonRenderNavigationUnit] | None = None


class AmazonRenderTocItem(BaseKindleModel):
    """Render archive raw TOC item with possible nested entries."""

    label: str
    toc_position_id: int
    entries: list["AmazonRenderTocItem"] | None = None


class ExtractionStatus(BaseKindleModel):
    """Status tracking for reader screen extraction and position restoration."""

    status: Literal["complete", "partial"] = "partial"
    reason: str | None = None
    restoration: Literal["succeeded", "failed", "unavailable"] | None = None


class Nav(BaseKindleModel):
    """Navigation range bounds for the book."""

    start_position: int = -1
    end_position: int = -1
    start_content_position: int = -1
    start_content_page: int = -1
    end_content_position: int = -1
    end_content_page: int = -1
    total_num_pages: int = -1
    total_num_content_pages: int = -1


class BookMetadata(BaseKindleModel):
    """Complete book extraction metadata document."""

    meta: AmazonBookMeta
    schema_version: int | None = Field(
        default=None,
        validation_alias=AliasChoices("schemaVersion", "schema_version"),
        serialization_alias="schemaVersion",
    )
    extraction: ExtractionStatus | None = None
    info: dict[str, JsonValue] | None = None
    nav: Nav = Field(default_factory=Nav)
    toc: list[TocItem] = Field(default_factory=list)
    pages: list[PageChunk] = Field(default_factory=list)
    location_map: AmazonRenderLocationMap | None = None


class TranscriptionReportPage(BaseKindleModel):
    """Per-page outcome entry in the transcription run report."""

    index: int
    hash: str | None = None
    status: str
    error: str | None = None


class TranscriptionIdentity(BaseKindleModel):
    """Cache and audit identity for a transcription model run."""

    provider: str
    model: str
    prompt_version: int = 1
    temperature: float = 0.0


class TranscriptionReport(BaseKindleModel):
    """Audit report for transcription run status and per-page outcomes."""

    status: Literal["running", "complete", "partial"]
    expected: int
    run_id: str
    identity: TranscriptionIdentity
    captures: list[PageChunk]
    content_hash: str | None = None
    pages: list[TranscriptionReportPage] = Field(default_factory=list)


class BookChapter(BaseKindleModel):
    """Structured chapter grouping a sequence of content chunks."""

    label: str
    depth: int
    chunks: list[ContentChunk]
