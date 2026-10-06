"""Tests for transcription provider adapters, pydantic-ai integration, and retry delay parsing."""

from unittest.mock import AsyncMock, MagicMock
import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ContentFilterError, ModelHTTPError
from pydantic_ai.models.test import TestModel

from python_kindle_ai_export.providers import (
    PydanticAiTranscriptionProvider,
    create_resilient_agent,
    create_transcription_provider,
    extract_retry_after,
)
from python_kindle_ai_export.transcription import TranscriptionError


def test_extract_retry_after_from_headers_and_body() -> None:
    headers = {"Retry-After": "4"}
    assert extract_retry_after(headers, None) == 4000

    body = {"error": {"details": [{"retryDelay": "8s"}]}}
    assert extract_retry_after(None, body) == 8000

    body_num = {"details": [{"retryDelay": "12"}]}
    assert extract_retry_after(None, body_num) == 12000

    assert extract_retry_after(None, None) is None


@pytest.mark.asyncio
async def test_normalizes_success_and_blank_page() -> None:
    agent_blank = Agent(TestModel(custom_output_text="<BLANK_PAGE>"))
    prov_blank = PydanticAiTranscriptionProvider(name="test", model="test:test", agent=agent_blank)
    res_blank = await prov_blank.transcribe(b"png")
    assert res_blank.status == "blank"
    assert res_blank.text == ""

    agent_success = Agent(TestModel(custom_output_text="Visible book text"))
    prov_success = PydanticAiTranscriptionProvider(name="test", model="test:test", agent=agent_success)
    res_success = await prov_success.transcribe(b"png")
    assert res_success.status == "success"
    assert res_success.text == "Visible book text"


@pytest.mark.asyncio
async def test_distinguishes_truncated_and_blocked_responses() -> None:
    agent_mock = MagicMock()

    # Truncated response
    res_trunc = MagicMock()
    res_trunc.output = "partial"
    res_trunc.response.finish_reason = "length"
    agent_mock.run = AsyncMock(return_value=res_trunc)

    prov = PydanticAiTranscriptionProvider(name="test", model="test:test", agent=agent_mock)
    with pytest.raises(TranscriptionError) as exc_info:
        await prov.transcribe(b"png")
    assert exc_info.value.kind == "truncated"

    # Blocked response
    res_blocked = MagicMock()
    res_blocked.output = "blocked"
    res_blocked.response.finish_reason = "safety"
    agent_mock.run = AsyncMock(return_value=res_blocked)

    with pytest.raises(TranscriptionError) as exc_info2:
        await prov.transcribe(b"png")
    assert exc_info2.value.kind == "blocked"


@pytest.mark.asyncio
async def test_content_filter_recitation_treated_as_blank_page() -> None:
    agent_mock = MagicMock()
    err_recitation = ContentFilterError("Content filter triggered. Finish reason: 'RECITATION', body: []")
    agent_mock.run = AsyncMock(side_effect=err_recitation)

    prov = PydanticAiTranscriptionProvider(name="test", model="test:test", agent=agent_mock)
    res = await prov.transcribe(b"png")
    assert res.status == "blank"
    assert res.text == ""


@pytest.mark.asyncio
async def test_content_filter_non_recitation_raises_blocked() -> None:
    agent_mock = MagicMock()
    err_safety = ContentFilterError("Content filter triggered. Finish reason: 'SAFETY', body: []")
    agent_mock.run = AsyncMock(side_effect=err_safety)

    prov = PydanticAiTranscriptionProvider(name="test", model="test:test", agent=agent_mock)
    with pytest.raises(TranscriptionError) as exc_info:
        await prov.transcribe(b"png")
    assert exc_info.value.kind == "blocked"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_retries_server_errors_and_rate_limits() -> None:
    agent_mock = MagicMock()
    # 429 rate limit
    err_429 = ModelHTTPError(status_code=429, model_name="test", headers={"retry-after": "5"})
    agent_mock.run = AsyncMock(side_effect=err_429)

    prov = PydanticAiTranscriptionProvider(name="test", model="test:test", agent=agent_mock)
    with pytest.raises(TranscriptionError) as exc_info:
        await prov.transcribe(b"png")
    assert exc_info.value.retryable is True
    assert exc_info.value.retry_after_ms == 5000

    # 401 unauthorized (fatal, not retryable)
    err_401 = ModelHTTPError(status_code=401, model_name="test")
    agent_mock.run = AsyncMock(side_effect=err_401)
    with pytest.raises(TranscriptionError) as exc_info2:
        await prov.transcribe(b"png")
    assert exc_info2.value.retryable is False
    assert exc_info2.value.fatal is True


def test_create_transcription_provider_normalizes_model_names() -> None:
    prov_openai = create_transcription_provider("openai", "gpt-4o-mini", "test-key")
    assert prov_openai.name == "openai"
    assert prov_openai.model == "openai:gpt-4o-mini"

    prov_gemini = create_transcription_provider("gemini", "gemini-2.5-flash", "test-key")
    assert prov_gemini.name == "gemini"
    assert prov_gemini.model == "google:gemini-2.5-flash"

    with pytest.raises(ValueError, match="Unsupported AI_PROVIDER"):
        create_transcription_provider("unsupported_provider", "model")


def test_universal_ai_key_mapping() -> None:
    import os
    from unittest.mock import patch
    from python_kindle_ai_export.providers import update_env_variables

    test_cases = [
        ("openai:gpt-4", "OPENAI_API_KEY"),
        ("anthropic:claude-3", "ANTHROPIC_API_KEY"),
        ("gemini:gemini-2.5-flash", "GOOGLE_API_KEY"),
        ("google:gemini-2.5-flash", "GOOGLE_API_KEY"),
        ("azure:gpt-4", "AZURE_OPENAI_API_KEY"),
        ("groq:llama3", "GROQ_API_KEY"),
    ]

    for model, target_var in test_cases:
        with patch.dict(
            os.environ,
            {"AI_KEY": "test-key-123", "MODEL": model},
            clear=True,
        ):
            update_env_variables()
            assert os.environ.get(target_var) == "test-key-123", f"Failed for model {model}"


def test_ai_key_precedence() -> None:
    import os
    from unittest.mock import patch
    from python_kindle_ai_export.providers import update_env_variables

    with patch.dict(
        os.environ,
        {
            "AI_KEY": "universal-key",
            "OPENAI_API_KEY": "existing-key",
            "MODEL": "openai:gpt-4",
        },
        clear=True,
    ):
        update_env_variables()
        assert os.environ.get("OPENAI_API_KEY") == "existing-key"


def test_create_transcription_provider_polymorphic_env_vars() -> None:
    import os
    from unittest.mock import patch

    with patch.dict(
        os.environ,
        {"MODEL": "google:gemini-2.5-flash", "AI_KEY": "gemini-test-key"},
        clear=True,
    ):
        prov = create_transcription_provider()
        assert prov.name == "gemini"
        assert prov.model == "google:gemini-2.5-flash"
        assert os.environ.get("GOOGLE_API_KEY") == "gemini-test-key"


def test_create_transcription_provider_infers_provider_from_bare_model() -> None:
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"AI_KEY": "dummy-key"}, clear=True):
        prov_gemini = create_transcription_provider(model="gemini-2.5-flash")
        assert prov_gemini.name == "gemini"
        assert prov_gemini.model == "google:gemini-2.5-flash"

        prov_claude = create_transcription_provider(model="claude-3-5-sonnet")
        assert prov_claude.name == "anthropic"
        assert prov_claude.model == "anthropic:claude-3-5-sonnet"


def test_create_transcription_provider_defaults_to_gemini_3_8_flash() -> None:
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"KINDLE_AI_EXPORT_AI_KEY": "test-key"}, clear=True):
        prov = create_transcription_provider()
        assert prov.name == "gemini"
        assert prov.model == "google:gemini-3.8-flash"
        assert os.environ.get("GOOGLE_API_KEY") == "test-key"
        assert os.environ.get("GEMINI_API_KEY") == "test-key"


def test_kindle_ai_export_prefix_precedence() -> None:
    import os
    from unittest.mock import patch
    from python_kindle_ai_export.utils import get_env

    with patch.dict(
        os.environ,
        {
            "KINDLE_AI_EXPORT_ASIN": "asin-prefixed",
            "ASIN": "asin-unprefixed",
        },
        clear=True,
    ):
        assert get_env("ASIN") == "asin-prefixed"
        assert get_env("KINDLE_AI_EXPORT_ASIN") == "asin-prefixed"


def test_create_resilient_agent_configures_sdk_retries() -> None:
    import os
    from unittest.mock import patch
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.models.anthropic import AnthropicModel

    with patch.dict(
        os.environ,
        {
            "GOOGLE_API_KEY": "test-key",
            "OPENAI_API_KEY": "test-key",
            "ANTHROPIC_API_KEY": "test-key",
            "KINDLE_AI_EXPORT_SDK_RETRIES": "12",
        },
        clear=True,
    ):
        google_agent = create_resilient_agent("google:gemini-3.8-flash")
        assert isinstance(google_agent.model, GoogleModel)

        openai_agent = create_resilient_agent("openai:gpt-4o-mini")
        assert isinstance(openai_agent.model, OpenAIChatModel)

        anthropic_agent = create_resilient_agent("anthropic:claude-3-5-haiku-latest")
        assert isinstance(anthropic_agent.model, AnthropicModel)

        # Fallback for plain/test models
        test_agent = create_resilient_agent("test")
        assert test_agent is not None



