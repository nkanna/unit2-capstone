"""Deprecated import shim. All calls now use Gemini; no AWS runtime remains."""
from llm_client import invoke, LLMResponse

BedrockResponse = LLMResponse  # legacy import compatibility only
