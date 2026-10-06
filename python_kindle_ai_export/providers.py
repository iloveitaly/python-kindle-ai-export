"""Transcription provider adapters using pydantic-ai."""

import os
import re
from collections.abc import Mapping
from pydantic_ai import Agent, BinaryContent, ModelSettings
from pydantic_ai.exceptions import ContentFilterError, ModelAPIError, ModelHTTPError, UserError

from .log import log
from .transcription import (
    TRANSCRIPTION_PROMPT,
    TranscriptionError,
    TranscriptionProvider,
    TranscriptionResult,
    parse_transcription,
)

RETRY_DELAY_REGEX = re.compile(r"^(\d+(?:\.\d+)?)\s*s?$", re.IGNORECASE)


def extract_retry_after(
    headers: Mapping[str, str] | None,
    body: object | None,
) -> int | None:
    """Extract retry delay milliseconds from HTTP headers or Gemini JSON error details.

    Pydantic-AI natively supports standard HTTP ``Retry-After`` headers via
    ``ModelHTTPError.retry_after`` (handling delta-seconds and RFC 9110 dates).

    However, this body extraction logic is specifically needed for Google Gemini. When
    Gemini encounters rate limits (HTTP 429 / RESOURCE_EXHAUSTED), Google does not
    return a standard ``Retry-After`` HTTP header. Instead, it embeds the retry window
    inside the response body in ``google.rpc.RetryInfo`` details (e.g. ``retryDelay: "18s"``).
    Because Pydantic-AI's ``ModelHTTPError.retry_after`` only inspects headers, it returns
    ``None`` on Gemini errors, necessitating this fallback.

    References:
    - https://github.com/pydantic/pydantic-ai/issues/5040 (Gemini 429 error details with retryDelay)
    - https://github.com/pydantic/pydantic-ai/issues/3728 (Rate-limit handling discussions)
    """
    if headers:
        for k, v in headers.items():
            if k.lower() == "retry-after":
                try:
                    return int(float(v) * 1000)
                except ValueError:
                    pass

    if isinstance(body, dict):
        error_dict = body.get("error")
        raw_details = body.get("details") or (
            error_dict.get("details") if isinstance(error_dict, dict) else None
        )
        if isinstance(raw_details, list):
            for item in raw_details:
                if isinstance(item, dict) and "retryDelay" in item:
                    delay_val = str(item["retryDelay"])
                    match = RETRY_DELAY_REGEX.match(delay_val)
                    if match:
                        return int(float(match.group(1)) * 1000)

    return None


DEFAULT_MODEL_NAME = "google:gemini-3.8-flash"

KNOWN_PROVIDERS = {
    "openai",
    "anthropic",
    "google",
    "gemini",
    "google-gla",
    "google-vertex",
    "groq",
    "mistral",
    "cohere",
    "azure",
    "ollama",
    "deepseek",
    "together",
    "bedrock",
    "test",
}


def map_ai_key(ai_key: str, model_name: str) -> None:
    """Map universal AI_KEY to provider-specific environment variables.

    Non-obvious behavior & design decisions:
    1. Model Prefix Inference: Automatically parses the target model string (e.g. 'google:gemini-3.8-flash',
       'gpt-4o-mini', 'claude-3-5-sonnet') to identify the target provider and set the appropriate
       native environment variable ('GOOGLE_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', etc.).
    2. Non-Destructive Mapping: Only sets target variables if they are not already present in the
       environment, preserving explicit user overrides.
    3. Gemini Dual-Naming Compatibility: Sets both ``GOOGLE_API_KEY`` and ``GEMINI_API_KEY`` to ensure
       compatibility across both legacy Google SDKs and newer google-genai clients.
    """
    provider = model_name.split(":")[0].lower() if ":" in model_name else ""
    if not provider:
        if model_name.startswith(("gpt-", "o1", "o3", "chatgpt")):
            provider = "openai"
        elif model_name.startswith("gemini-"):
            provider = "google"
        elif model_name.startswith("claude-"):
            provider = "anthropic"
        elif model_name.startswith("deepseek-"):
            provider = "deepseek"
        else:
            provider = "openai"

    mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "google-gla": "GOOGLE_API_KEY",
        "google-vertex": "GOOGLE_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "cohere": "CO_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "together": "TOGETHER_API_KEY",
        "ollama": "OLLAMA_API_KEY",
    }
    target_key = mapping.get(provider)
    if target_key and target_key not in os.environ:
        os.environ[target_key] = ai_key

    if provider in ("google", "gemini", "google-gla", "google-vertex") and "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = ai_key


def update_env_variables() -> None:
    """Sync prefixed variables and universal keys to standard provider vars.

    Resolves prefixed configuration across projects using the priority chain:
    ``KINDLE_AI_EXPORT_*`` -> ``KINDLE_EXPORT_*`` -> ``AIAUTOCOMMIT_*`` -> base variables.
    Also ensures bidirectional synchronization between ``GOOGLE_API_KEY`` and ``GEMINI_API_KEY``.
    """
    prefixes = ("KINDLE_AI_EXPORT_", "KINDLE_EXPORT_", "AIAUTOCOMMIT_")
    for key in list(os.environ.keys()):
        for prefix in prefixes:
            if key.startswith(prefix):
                base_key = key[len(prefix) :]
                os.environ[base_key] = os.environ[key]

    if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

    ai_key = (
        os.environ.get("KINDLE_AI_EXPORT_AI_KEY")
        or os.environ.get("AI_KEY")
        or os.environ.get("KINDLE_EXPORT_AI_KEY")
    )
    if ai_key:
        model_name = (
            os.environ.get("KINDLE_AI_EXPORT_MODEL")
            or os.environ.get("MODEL")
            or os.environ.get("KINDLE_EXPORT_MODEL")
            or os.environ.get("AI_MODEL")
            or os.environ.get("GEMINI_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or DEFAULT_MODEL_NAME
        )
        map_ai_key(ai_key, model_name)


def create_resilient_agent(target_model: str, api_key: str | None = None) -> Agent:
    """Instantiate a Pydantic-AI Agent with customized transport-level SDK retries.

    Two-Tier Retry Architecture:
    - Tier 1 (Transport Layer / This Function): Configures provider SDK clients (Google GenAI,
      OpenAI, Anthropic) with aggressive short-term retries (default 10 attempts) and backoff
      caps (up to 120s) for HTTP 408, 429, 500, 502, 503, and 504 status codes. Transient spikes
      are resolved invisibly inside ``agent.run()`` without failing or needing outer orchestration.
    - Tier 2 (Application Layer / ``with_retries``): When SDK-level retries are exhausted,
      application-level exponential backoff steps in to handle multi-hour rate limits, quota
      depletion, and checkpoint saving across up to 100 attempts.
    """
    sdk_retries = int(
        os.environ.get("KINDLE_AI_EXPORT_SDK_RETRIES")
        or os.environ.get("SDK_RETRIES")
        or "10"
    )
    prefix, sep, model_part = target_model.partition(":")
    if not sep or not model_part:
        return Agent(target_model)

    prefix_lower = prefix.lower()
    if prefix_lower in ("google", "gemini"):
        try:
            from google.genai.types import HttpRetryOptions
            from pydantic_ai.providers.google import GoogleProvider
            from pydantic_ai.models.google import GoogleModel

            key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if key:
                retry_opts = HttpRetryOptions(
                    attempts=sdk_retries,
                    initial_delay=2.0,
                    max_delay=120.0,
                    exp_base=2.0,
                    jitter=1.0,
                    http_status_codes=[408, 429, 500, 502, 503, 504],
                )
                provider = GoogleProvider(api_key=key, retry_options=retry_opts)
                return Agent(GoogleModel(model_part, provider=provider))
        except (ImportError, Exception):
            pass

    elif prefix_lower == "openai":
        try:
            from openai import AsyncOpenAI
            from pydantic_ai.providers.openai import OpenAIProvider
            from pydantic_ai.models.openai import OpenAIChatModel

            key = api_key or os.getenv("OPENAI_API_KEY") or "placeholder"
            base_url = os.getenv("OPENAI_BASE_URL")
            client = AsyncOpenAI(api_key=key, base_url=base_url, max_retries=sdk_retries)
            provider = OpenAIProvider(openai_client=client)
            return Agent(OpenAIChatModel(model_part, provider=provider))
        except (ImportError, Exception):
            pass

    elif prefix_lower == "anthropic":
        try:
            from anthropic import AsyncAnthropic
            from pydantic_ai.providers.anthropic import AnthropicProvider
            from pydantic_ai.models.anthropic import AnthropicModel

            key = api_key or os.getenv("ANTHROPIC_API_KEY") or "placeholder"
            client = AsyncAnthropic(api_key=key, max_retries=sdk_retries)
            provider = AnthropicProvider(anthropic_client=client)
            return Agent(AnthropicModel(model_part, provider=provider))
        except (ImportError, Exception):
            pass

    return Agent(target_model)


class PydanticAiTranscriptionProvider:
    """Transcription provider adapter powered by pydantic-ai."""

    def __init__(
        self,
        name: str,
        model: str,
        agent: Agent | None = None,
        api_key: str | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.agent = agent or create_resilient_agent(model, api_key=api_key)

    async def transcribe(self, image: bytes) -> TranscriptionResult:
        """Submit page screenshot to model and parse verbatim text or blank marker.

        Non-obvious behavior & design decisions:
        1. Gemini Thinking Budget: Sets ``ModelSettings(thinking="low")`` for Google Gemini models.
           Gemini 2.5/3.x models enable thinking by default, which expends unnecessary reasoning tokens
           and adds latency for verbatim OCR tasks. Lowering thinking minimizes token overhead.
        2. Non-Gemini Temperature: Uses ``temperature=0.0`` for other providers to ensure deterministic,
           verbatim text reproduction without creative hallucination.
        3. Finish Reason Classification:
           - ``CONTENT_FILTER`` / ``SAFETY``: Classified as non-retryable ``blocked`` failure.
           - ``LENGTH`` / ``MAX_TOKENS``: Classified as retryable ``truncated`` failure.
           - Non-standard incomplete finish reasons: Also classified as retryable ``truncated``.
        4. Error Normalization:
           - 401 (unauthorized), 403 (forbidden), 404 (nonexistent model) are classified as fatal.
           - 429 (rate limit) and other HTTP errors are classified as retryable, extracting wait delays.
        """
        model_settings: ModelSettings | None = None
        from pydantic_ai.models.google import GoogleModel

        if isinstance(self.agent.model, GoogleModel):
            # Google Gemini models accept thinking="low"
            model_settings = ModelSettings(thinking="low")
        else:
            model_settings = ModelSettings(temperature=0.0)

        try:
            result = await self.agent.run(
                [
                    TRANSCRIPTION_PROMPT,
                    BinaryContent(data=image, media_type="image/png"),
                ],
                model_settings=model_settings,
            )
        except ModelHTTPError as err:
            status = err.status_code
            if err.retry_after is not None:
                retry_after = int(err.retry_after * 1000)
            else:
                retry_after = extract_retry_after(err.headers, err.body)
            # 401 (invalid key), 403 (unauthorized/forbidden), 404 (nonexistent model) are fatal
            is_fatal = status in (401, 403, 404)
            is_retryable = not is_fatal
            raise TranscriptionError(
                f"{self.name} request failed (HTTP {status})",
                kind="failed",
                retryable=is_retryable,
                retry_after_ms=retry_after,
                fatal=is_fatal,
            ) from err
        except ModelAPIError as err:
            retry_after = extract_retry_after(None, getattr(err, "body", None))
            raise TranscriptionError(
                f"{self.name} API error: {err}",
                kind="failed",
                retryable=True,
                retry_after_ms=retry_after,
                fatal=False,
            ) from err
        except UserError as err:
            raise TranscriptionError(
                f"{self.name} configuration error: {err}",
                kind="failed",
                retryable=False,
                fatal=True,
            ) from err
        except ContentFilterError as err:
            err_str = str(err)
            if "RECITATION" in err_str:
                log.warning("page triggered recitation filter; recording as blank page", error=err_str)
                return TranscriptionResult(status="blank", text="")
            raise TranscriptionError(
                f"{self.name} blocked the page: {err}",
                kind="blocked",
                retryable=False,
            ) from err

        finish_reason = getattr(result.response, "finish_reason", None)
        if finish_reason:
            reason_str = str(finish_reason).upper()
            if reason_str in ("CONTENT_FILTER", "SAFETY"):
                raise TranscriptionError(
                    f"{self.name} blocked the page",
                    kind="blocked",
                    retryable=False,
                )
            if reason_str in ("LENGTH", "MAX_TOKENS"):
                raise TranscriptionError(
                    f"{self.name} truncated the page",
                    kind="truncated",
                    retryable=True,
                )
            if reason_str not in ("STOP", "END_TURN"):
                raise TranscriptionError(
                    f"{self.name} response was not complete",
                    kind="truncated",
                    retryable=True,
                )

        return parse_transcription(result.output)


def create_transcription_provider(
    name: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> TranscriptionProvider:
    """Create a configured TranscriptionProvider instance for the given AI provider and model.

    Polymorphically supports any provider via Pydantic-AI (e.g. openai, google/gemini, anthropic, groq, etc.).
    Automatically infers provider and model from MODEL and AI_KEY environment variables if not passed directly.
    """
    update_env_variables()

    raw_model = (
        model
        or os.environ.get("KINDLE_AI_EXPORT_MODEL")
        or os.environ.get("MODEL")
        or os.environ.get("KINDLE_EXPORT_MODEL")
        or os.environ.get("AI_MODEL")
    )
    if not raw_model:
        if name:
            norm_name = name.lower()
            if norm_name in ("gemini", "google"):
                raw_model = os.environ.get("KINDLE_AI_EXPORT_GEMINI_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-3.8-flash"
            elif norm_name == "anthropic":
                raw_model = os.environ.get("KINDLE_AI_EXPORT_ANTHROPIC_MODEL") or os.environ.get("ANTHROPIC_MODEL") or "claude-3-5-haiku-latest"
            else:
                raw_model = os.environ.get("KINDLE_AI_EXPORT_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        else:
            raw_model = DEFAULT_MODEL_NAME

    # Resolve target model string and provider name
    if ":" in raw_model:
        prefix, model_part = raw_model.split(":", 1)
        prefix_lower = prefix.lower()
        if prefix_lower in ("gemini", "google"):
            target_model = f"google:{model_part}"
            provider_name = name.lower() if name else "gemini"
        else:
            target_model = raw_model
            provider_name = name.lower() if name else prefix_lower
    else:
        if name:
            provider_name = name.lower()
            if provider_name in ("gemini", "google"):
                target_model = f"google:{raw_model}"
            else:
                target_model = f"{provider_name}:{raw_model}"
        else:
            if raw_model.startswith(("gpt-", "o1", "o3", "chatgpt")):
                provider_name = "openai"
                target_model = f"openai:{raw_model}"
            elif raw_model.startswith("gemini-"):
                provider_name = "gemini"
                target_model = f"google:{raw_model}"
            elif raw_model.startswith("claude-"):
                provider_name = "anthropic"
                target_model = f"anthropic:{raw_model}"
            elif raw_model.startswith("deepseek-"):
                provider_name = "deepseek"
                target_model = f"deepseek:{raw_model}"
            else:
                provider_name = "openai"
                target_model = f"openai:{raw_model}"

    if provider_name not in KNOWN_PROVIDERS:
        raise ValueError(f"Unsupported AI_PROVIDER: {name or provider_name}")

    resolved_key = (
        api_key
        or os.environ.get("AI_KEY")
        or os.environ.get("KINDLE_AI_EXPORT_AI_KEY")
        or os.environ.get("KINDLE_EXPORT_AI_KEY")
    )
    if resolved_key:
        map_ai_key(resolved_key, target_model)

    return PydanticAiTranscriptionProvider(
        name=provider_name,
        model=target_model,
        api_key=resolved_key,
    )
