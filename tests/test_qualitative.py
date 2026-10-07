"""
Offline unit tests for agents/qualitative.py.

All tests use mocking — no Bedrock calls, no ChromaDB access.
Every assertion tests real code paths, not fabricated outputs.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from agents.qualitative import QualitativeAgent, QualitativeResult
from validation.validator import ValidationResult


# ── Helpers ───────────────────────────────────────────────────────────────────

@dataclass
class _FakeBedrockResponse:
    text: str
    input_tokens: int = 100
    output_tokens: int = 80
    stop_reason: str = "end_turn"


def _make_valid_json(
    answer="Passwords must be at least 16 characters.",
    chunk_id="security_policy.md::chunk_1",
    refused=False,
) -> str:
    return json.dumps({
        "answer": answer,
        "citations": [
            {
                "chunk_id": chunk_id,
                "source": chunk_id.split("::")[0],
                "excerpt": "Passwords must be at least 16 characters",
            }
        ],
        "refused": refused,
        "refusal_reason": None,
    })


def _make_chunks(distances: list[float], source: str = "security_policy.md") -> list[dict]:
    # Document text includes the standard test excerpt so excerpt validation passes.
    return [
        {
            "id": f"{source}::chunk_{i}",
            "document": f"Chunk {i}: Passwords must be at least 16 characters in length.",
            "distance": d,
            "source": source,
            "chunk_id": i,
        }
        for i, d in enumerate(distances)
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestQualitativeAgentUnit:

    def _agent_with_mock_retrieve(self, chunks: list[dict]) -> QualitativeAgent:
        agent = QualitativeAgent.__new__(QualitativeAgent)
        agent._chroma_path = None
        agent._collection_name = None
        agent._collection = MagicMock()
        agent._retrieve = MagicMock(return_value=chunks)
        return agent

    # ── Happy path ─────────────────────────────────────────────────────────

    def test_valid_answer_returned(self, mocker):
        """Agent returns parsed answer when Claude response is valid."""
        chunks = _make_chunks([0.25, 0.40, 0.55, 0.60])
        agent = self._agent_with_mock_retrieve(chunks)

        valid_json = _make_valid_json(chunk_id="security_policy.md::chunk_0")
        mock_resp = _FakeBedrockResponse(text=valid_json)

        mocker.patch("agents.qualitative.bc.invoke", return_value=mock_resp)
        mocker.patch("agents.qualitative.log_call", return_value=MagicMock())

        result = agent.answer("What are the password rules?", "test-qid")

        assert not result.refused
        assert "16 characters" in result.answer
        assert result.attempts == 1
        assert result.validation.valid

    def test_valid_citations_returned(self, mocker):
        """Citation chunk_id must match a retrieved chunk."""
        chunks = _make_chunks([0.30])
        agent = self._agent_with_mock_retrieve(chunks)

        valid_json = _make_valid_json(chunk_id="security_policy.md::chunk_0")
        mocker.patch("agents.qualitative.bc.invoke",
                     return_value=_FakeBedrockResponse(text=valid_json))
        mocker.patch("agents.qualitative.log_call", return_value=MagicMock())

        result = agent.answer("Password policy?", "test-qid")

        assert len(result.citations) == 1
        assert result.citations[0]["chunk_id"] == "security_policy.md::chunk_0"

    # ── Unsupported question (pre-filter) ──────────────────────────────────

    def test_unsupported_question_refuses_without_calling_claude(self, mocker):
        """When all distances exceed the threshold, refuse before calling Bedrock."""
        chunks = _make_chunks([0.72, 0.80, 0.85, 0.90])
        agent = self._agent_with_mock_retrieve(chunks)

        mock_invoke = mocker.patch("agents.qualitative.bc.invoke")

        result = agent.answer("What is the coffee machine policy?", "test-qid")

        assert result.refused
        assert result.attempts == 0
        mock_invoke.assert_not_called()

    # ── Fabricated citation ────────────────────────────────────────────────

    def test_fabricated_citation_triggers_retry(self, mocker):
        """If Claude returns a chunk_id outside the retrieved set, retry."""
        chunks = _make_chunks([0.30, 0.40])
        agent = self._agent_with_mock_retrieve(chunks)

        # First call: fabricated citation; second call: valid
        bad_json = _make_valid_json(chunk_id="completely_made_up.md::chunk_99")
        good_json = _make_valid_json(chunk_id="security_policy.md::chunk_0")

        mocker.patch(
            "agents.qualitative.bc.invoke",
            side_effect=[
                _FakeBedrockResponse(text=bad_json),
                _FakeBedrockResponse(text=good_json),
            ],
        )
        mocker.patch("agents.qualitative.log_call", return_value=MagicMock())

        result = agent.answer("Password requirements?", "test-qid")

        assert result.attempts == 2
        assert result.validation.valid
        assert result.citations[0]["chunk_id"] == "security_policy.md::chunk_0"

    # ── Malformed JSON ─────────────────────────────────────────────────────

    def test_malformed_json_triggers_retry_then_fails(self, mocker):
        """Two consecutive malformed JSON responses exhaust retries → refused."""
        chunks = _make_chunks([0.30])
        agent = self._agent_with_mock_retrieve(chunks)

        mocker.patch(
            "agents.qualitative.bc.invoke",
            side_effect=[
                _FakeBedrockResponse(text="not json at all }{"),
                _FakeBedrockResponse(text="still broken json }{"),
            ],
        )
        mocker.patch("agents.qualitative.log_call", return_value=MagicMock())

        result = agent.answer("What is the policy?", "test-qid")

        assert result.refused
        assert result.attempts == 2
        assert not result.validation.valid

    # ── Truncated response ─────────────────────────────────────────────────

    def test_truncated_response_triggers_retry(self, mocker):
        """stop_reason='max_tokens' (truncated) should trigger a retry."""
        chunks = _make_chunks([0.28])
        agent = self._agent_with_mock_retrieve(chunks)

        # First: truncated; second: valid full response
        truncated = _FakeBedrockResponse(
            text=_make_valid_json(chunk_id="security_policy.md::chunk_0"),
            stop_reason="max_tokens",
        )
        complete = _FakeBedrockResponse(
            text=_make_valid_json(chunk_id="security_policy.md::chunk_0"),
            stop_reason="end_turn",
        )

        mocker.patch(
            "agents.qualitative.bc.invoke",
            side_effect=[truncated, complete],
        )
        mocker.patch("agents.qualitative.log_call", return_value=MagicMock())

        result = agent.answer("Security policy?", "test-qid")

        assert result.attempts == 2
        assert result.validation.valid

    # ── Claude refuses (content-based) ────────────────────────────────────

    def test_claude_content_refusal_accepted(self, mocker):
        """Claude may refuse even when chunks pass the distance pre-filter."""
        chunks = _make_chunks([0.56, 0.60])  # below threshold but low relevance
        agent = self._agent_with_mock_retrieve(chunks)

        refused_json = json.dumps({
            "answer": "",
            "citations": [],
            "refused": True,
            "refusal_reason": "The context does not contain vacation policy information.",
        })
        mocker.patch(
            "agents.qualitative.bc.invoke",
            return_value=_FakeBedrockResponse(text=refused_json),
        )
        mocker.patch("agents.qualitative.log_call", return_value=MagicMock())

        result = agent.answer("What is the vacation policy?", "test-qid")

        assert result.refused
        assert result.validation.valid   # a clean refusal is valid
        assert "vacation" in result.refusal_reason.lower()

    # ── Conversation history ───────────────────────────────────────────────

    def test_history_is_prepended_to_messages(self, mocker):
        """Prior conversation turns must be sent before the new question."""
        chunks = _make_chunks([0.30])
        agent = self._agent_with_mock_retrieve(chunks)

        good_json = _make_valid_json(chunk_id="security_policy.md::chunk_0")
        mock_invoke = mocker.patch(
            "agents.qualitative.bc.invoke",
            return_value=_FakeBedrockResponse(text=good_json),
        )
        mocker.patch("agents.qualitative.log_call", return_value=MagicMock())

        history = [
            {"role": "user", "content": "Tell me about security."},
            {"role": "assistant", "content": "Sure, here is what I know."},
        ]
        agent.answer("What about MFA?", "test-qid", history=history)

        call_args = mock_invoke.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        # First two messages must be the history
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Tell me about security."
        assert messages[1]["role"] == "assistant"

    # ── Raw evidence is always returned ───────────────────────────────────

    def test_retrieved_chunks_always_returned_for_human_review(self, mocker):
        """retrieved_chunks must be present even on refusal — for human review."""
        chunks = _make_chunks([0.50, 0.65])
        agent = self._agent_with_mock_retrieve(chunks)

        refused_json = json.dumps({
            "answer": "",
            "citations": [],
            "refused": True,
            "refusal_reason": "Not in context.",
        })
        mocker.patch(
            "agents.qualitative.bc.invoke",
            return_value=_FakeBedrockResponse(text=refused_json),
        )
        mocker.patch("agents.qualitative.log_call", return_value=MagicMock())

        result = agent.answer("Unrelated question?", "test-qid")

        assert len(result.retrieved_chunks) == 2
        assert result.retrieved_chunks[0]["distance"] == 0.50
