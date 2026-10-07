"""Shared JSON parsing utilities."""
import json
import re


def parse_json_object(text: str) -> dict:
    """
    Parse a JSON object from a Bedrock response.

    Accepts:
    - Plain JSON: {"key": "value"}
    - Single JSON code-fenced block: ```json\\n{...}\\n```

    Rejects with ValueError:
    - Surrounding prose outside the fenced block
    - Multiple code-fenced blocks
    - Valid JSON that is not a top-level object (list, string, etc.)
    - Malformed JSON
    - Empty string

    This function does NOT use eval() or exec().
    """
    if not text or not text.strip():
        raise ValueError("Response text is empty")

    stripped = text.strip()

    # Detect code fence: optional ```json or ``` then newline then content then ```
    fence_pattern = re.compile(
        r'^```(?:json)?\s*\n(.*?)\n```\s*$',
        re.DOTALL,
    )
    fence_match = fence_pattern.match(stripped)

    if fence_match:
        inner = fence_match.group(1).strip()
        try:
            parsed = json.loads(inner)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON inside code fence: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
        return parsed

    # Reject multiple fenced blocks
    if stripped.count("```") >= 4:
        raise ValueError(
            "Multiple code-fenced blocks detected; expected exactly one JSON object"
        )

    # No fence — text must start with '{' to be a plain JSON object
    if not stripped.startswith("{"):
        raise ValueError(
            f"Response is not a JSON object and is not code-fenced. "
            f"Starts with: {stripped[:60]!r}"
        )

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")

    return parsed
