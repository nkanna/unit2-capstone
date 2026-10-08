"""Shared Gemini client. The OpenAI SDK is a transport to Google's endpoint.

The existing invoke signature and end_turn/max_tokens validator contract are
preserved. Missing usage stays None; unsuccessful responses are never marked
complete. API error bodies and credentials are deliberately not surfaced.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.6-flash"
QUOTA_FALLBACK_MODEL = "gemini-3.5-flash-lite"
MAX_API_ATTEMPTS = 3
REQUEST_TIMEOUT_SECONDS = 90.0
_client = None


class LLMAPIError(RuntimeError):
    """Safe diagnostic without provider error bodies, URLs, or secrets."""

    def __init__(self, message: str, status_code: int | None = None,
                 attempts: list[dict] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.attempts = list(attempts or [])


@dataclass
class LLMResponse:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    stop_reason: str
    model: str = DEFAULT_MODEL
    provider_finish_reason: str | None = None
    total_tokens: int | None = None
    usage_metadata: dict = field(default_factory=dict)
    response_issue: str | None = None
    provider: str = "google_gemini"
    requested_model: str = DEFAULT_MODEL
    model_reported: bool = True
    attempts: list[dict] = field(default_factory=list)


def _get_client():
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key or key.startswith("<") or key == "your_gemini_api_key_here":
            raise LLMAPIError("GEMINI_API_KEY is missing or still a placeholder; configure it locally.")
        # Lazy import lets all application mocks run offline without an SDK/key.
        from openai import OpenAI

        _client = OpenAI(api_key=key, base_url=BASE_URL, max_retries=0,
                         timeout=REQUEST_TIMEOUT_SECONDS)
    return _client


def _dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _map_response(completion: Any, requested_model: str) -> LLMResponse:
    payload = _dict(completion)
    usage = _dict(payload.get("usage"))
    choices = payload.get("choices") or []
    choice = _dict(choices[0]) if choices else {}
    raw_reason = choice.get("finish_reason") or choice.get("finishReason")
    reason = str(raw_reason) if raw_reason is not None else None
    normalized = {
        "stop": "end_turn", "STOP": "end_turn",
        "length": "max_tokens", "MAX_TOKENS": "max_tokens",
        "content_filter": "safety_block", "SAFETY": "safety_block",
        "RECITATION": "safety_block", "BLOCKLIST": "safety_block",
        "PROHIBITED_CONTENT": "safety_block", "SPII": "safety_block",
    }.get(reason, reason or "missing_finish_reason")
    message = _dict(choice.get("message"))
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    feedback = _dict(payload.get("prompt_feedback") or payload.get("promptFeedback"))
    blocked = feedback.get("blockReason") or feedback.get("block_reason")
    issue = None
    if normalized == "safety_block" or blocked or message.get("refusal"):
        issue = "safety_block"
        normalized = "safety_block"
        text = ""  # blocked content must not reach downstream JSON/SQL execution
    elif normalized == "max_tokens":
        issue = "truncated_response"  # retain partial text and real usage
    elif not text.strip():
        issue = "missing_content"
        normalized = "missing_content"  # even stop + empty content is invalid
    elif normalized != "end_turn":
        issue = "incomplete_response"

    model = payload.get("model")
    return LLMResponse(
        text=text,
        input_tokens=_count(usage.get("prompt_tokens", usage.get("promptTokens"))),
        output_tokens=_count(usage.get("completion_tokens", usage.get("completionTokens"))),
        total_tokens=_count(usage.get("total_tokens", usage.get("totalTokens"))),
        stop_reason=normalized,
        provider_finish_reason=reason,
        model=model if isinstance(model, str) and model else requested_model,
        model_reported=bool(model), requested_model=requested_model,
        usage_metadata=usage, response_issue=issue,
    )


def invoke(messages: list[dict], system: str = "", max_tokens: int = 2048,
           temperature: float = 0.0) -> LLMResponse:
    """Invoke Gemini with unchanged system instructions and ordered history.

    HTTP 429, 503 and SDK timeouts are retried, with at most three requests and waits
    of 1s and 2s. Two quota failures select the approved fallback; a 503 alone
    retries the same model. Failed requests have no fabricated usage counts.

    temperature remains in the shared interface for existing callers, but is
    not sent: Gemini 3.6 Flash and 3.5 Flash-Lite ignore this deprecated option.
    """
    model = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    if model.startswith("<"):
        model = DEFAULT_MODEL
    formatted = ([{"role": "system", "content": system}] if system else []) + [dict(m) for m in messages]
    client = _get_client()
    attempts = []
    quota_failures = 0
    for attempt in range(MAX_API_ATTEMPTS):
        selected = QUOTA_FALLBACK_MODEL if quota_failures >= 2 else model
        try:
            completion = client.chat.completions.create(
                model=selected, messages=formatted, max_tokens=max_tokens,
            )
        except Exception as exc:
            from openai import APITimeoutError

            status = getattr(exc, "status_code", None)
            timed_out = isinstance(exc, APITimeoutError)
            attempts.append({"model": selected, "status_code": status, "outcome": "api_failure",
                             "error_type": "timeout" if timed_out else "api_error"})
            if status == 429:
                quota_failures += 1
            if (status in {429, 503} or timed_out) and attempt < MAX_API_ATTEMPTS - 1:
                time.sleep(2 ** attempt)
                continue
            detail = f"HTTP {status}" if status is not None else type(exc).__name__
            raise LLMAPIError(f"Gemini request failed ({detail}); {len(attempts)} attempt(s).", status, attempts) from None
        result = _map_response(completion, selected)
        attempts.append({"model": selected, "outcome": "response"})
        result.attempts = attempts
        return result
    raise AssertionError("Unreachable")
