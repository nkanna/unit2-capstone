"""
Shared Bedrock client for all LLM calls in this project.

AWS authentication is handled entirely by boto3 credential discovery
(~/.aws/credentials / ~/.aws/config set up with `aws configure`).
No API key is stored in this project.
"""

import json
import os
from dataclasses import dataclass

import boto3
from dotenv import load_dotenv

load_dotenv()

_AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")

# Lazily created so import-time failures are easy to diagnose
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=_AWS_REGION)
    return _client


@dataclass
class BedrockResponse:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str


def invoke(
    messages: list[dict],
    system: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> BedrockResponse:
    """
    Call Claude via Bedrock and return text, token counts, and stop_reason.

    messages: list of {"role": "user"|"assistant", "content": "..."} dicts
    system:   optional system prompt string
    """
    body: dict = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        body["system"] = system

    response = _get_client().invoke_model(
        modelId=_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    payload = json.loads(response["body"].read())

    text = "".join(
        block["text"] for block in payload["content"] if block["type"] == "text"
    )
    usage = payload.get("usage", {})

    return BedrockResponse(
        text=text,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        stop_reason=payload.get("stop_reason", ""),
    )
