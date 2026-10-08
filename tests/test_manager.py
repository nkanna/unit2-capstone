"""
Offline unit tests for agents/manager.py.

All tests use mocking — no Gemini calls, no ChromaDB, no SQLite.
Every assertion tests real code paths; nothing is fabricated.
"""

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from agents.manager import Manager, ManagerResult, SessionHistory, Turn
from validation.validator import ValidationResult


# ── Helpers ───────────────────────────────────────────────────────────────────

@dataclass
class _FakeLLMResponse:
    text: str
    input_tokens: int = 40
    output_tokens: int = 50
    stop_reason: str = "end_turn"


def _class_json(
    route: str = "qualitative",
    qual_sub: str | None = "What is the password policy?",
    quant_sub: str | None = None,
    reasoning: str = "policy question",
) -> str:
    return json.dumps({
        "route": route,
        "qualitative_subquery": qual_sub,
        "quantitative_subquery": quant_sub,
        "reasoning": reasoning,
    })


def _synth_json(synthesis: str = "Combined answer.", refused: bool = False) -> str:
    return json.dumps({
        "synthesis": synthesis,
        "refused": refused,
        "refusal_reason": None,
    })


def _make_qual_result(answer: str = "Policy answer.", refused: bool = False) -> MagicMock:
    r = MagicMock()
    r.answer = answer
    r.refused = refused
    r.refusal_reason = "No context." if refused else None
    r.citations = [{"chunk_id": "policy.md::chunk_0", "excerpt": "text"}]
    r.retrieved_chunks = [{"document": "Policy text with no relevant numbers."}]
    r.call_records = []
    r.validation = ValidationResult(valid=True)
    return r


def _make_quant_result(
    interpretation: str = "Revenue was $5,000.", refused: bool = False
) -> MagicMock:
    r = MagicMock()
    r.interpretation = interpretation
    r.refused = refused
    r.refusal_reason = "Blocked." if refused else None
    r.sql = "SELECT SUM(amount) FROM sales"
    r.col_names = ["total"]
    r.result_rows = [{"total": 5000.0}]
    r.row_count = 1
    r.call_records = []
    r.validation = ValidationResult(valid=True)
    return r


def _make_manager(mocker) -> Manager:
    """Create a Manager with mocked sub-agents and no real connections."""
    mgr = Manager.__new__(Manager)
    mgr._qual_agent = MagicMock()
    mgr._quant_agent = MagicMock()
    mgr._history = SessionHistory()
    return mgr


# ── SessionHistory ─────────────────────────────────────────────────────────────

class TestSessionHistory:

    def test_empty_history_returns_empty_messages(self):
        h = SessionHistory()
        assert h.to_messages() == []

    def test_add_turn_appears_in_messages(self):
        h = SessionHistory()
        h.add(Turn("What is the policy?", "It is 16 chars.", "qid-1", "qualitative"))
        msgs = h.to_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "What is the policy?"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "It is 16 chars."

    def test_reset_clears_history(self):
        h = SessionHistory()
        h.add(Turn("Q", "A", "qid-1", "qualitative"))
        h.reset()
        assert h.to_messages() == []
        assert len(h) == 0

    def test_history_bounded_by_max_turns(self):
        h = SessionHistory(max_turns=2)
        for i in range(5):
            h.add(Turn(f"Q{i}", f"A{i}", f"qid-{i}", "qualitative"))
        msgs = h.to_messages()
        # Only last 2 turns = 4 messages
        assert len(msgs) == 4
        assert msgs[0]["content"] == "Q3"
        assert msgs[2]["content"] == "Q4"


# ── Manager: routing ──────────────────────────────────────────────────────────

class TestManagerRouting:

    def test_qualitative_route(self, mocker):
        """Qualitative route calls only the qualitative agent."""
        mgr = _make_manager(mocker)
        qual_result = _make_qual_result()
        mgr._qual_agent.answer.return_value = qual_result

        mocker.patch("agents.manager.lc.invoke",
                     return_value=_FakeLLMResponse(text=_class_json(route="qualitative")))
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", return_value="test-qid-1")

        result = mgr.answer("What is the password policy?")

        mgr._qual_agent.answer.assert_called_once()
        mgr._quant_agent.answer.assert_not_called()
        assert result.classification == "qualitative"
        assert result.final_answer == "Policy answer."

    def test_quantitative_route(self, mocker):
        """Quantitative route calls only the quantitative agent."""
        mgr = _make_manager(mocker)
        quant_result = _make_quant_result()
        mgr._quant_agent.answer.return_value = quant_result

        mocker.patch(
            "agents.manager.lc.invoke",
            return_value=_FakeLLMResponse(text=_class_json(
                route="quantitative",
                qual_sub=None,
                quant_sub="What was total revenue?",
            )),
        )
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", return_value="test-qid-2")

        result = mgr.answer("What was total revenue?")

        mgr._quant_agent.answer.assert_called_once()
        mgr._qual_agent.answer.assert_not_called()
        assert result.classification == "quantitative"
        assert "Revenue" in result.final_answer

    def test_both_route_calls_both_agents_and_synthesizes(self, mocker):
        """'both' route calls both agents and invokes synthesis."""
        mgr = _make_manager(mocker)
        mgr._qual_agent.answer.return_value = _make_qual_result()
        mgr._quant_agent.answer.return_value = _make_quant_result()

        mock_invoke = mocker.patch(
            "agents.manager.lc.invoke",
            side_effect=[
                # 1st call: classification → both
                _FakeLLMResponse(text=_class_json(
                    route="both",
                    qual_sub="Remote work policy?",
                    quant_sub="How many remote employees?",
                )),
                # 2nd call: synthesis
                _FakeLLMResponse(text=_synth_json("According to policy... The data shows...")),
            ],
        )
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", return_value="test-qid-3")

        result = mgr.answer("Remote work policy and how many employees work remotely?")

        assert result.classification == "both"
        assert "According to policy" in result.final_answer or "Data" in result.final_answer
        mgr._qual_agent.answer.assert_called_once()
        mgr._quant_agent.answer.assert_called_once()

    def test_unsupported_route_refuses_without_agents(self, mocker):
        """'unsupported' route returns refusal without calling either agent."""
        mgr = _make_manager(mocker)

        mocker.patch(
            "agents.manager.lc.invoke",
            return_value=_FakeLLMResponse(text=_class_json(
                route="unsupported", qual_sub=None, quant_sub=None,
                reasoning="general world knowledge",
            )),
        )
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", return_value="test-qid-4")

        result = mgr.answer("What is the capital of France?")

        assert result.refused
        assert result.classification == "unsupported"
        mgr._qual_agent.answer.assert_not_called()
        mgr._quant_agent.answer.assert_not_called()


# ── Manager: history ──────────────────────────────────────────────────────────

class TestManagerHistory:

    def test_history_accumulates_across_turns(self, mocker):
        """Each successful turn is added to the session history."""
        mgr = _make_manager(mocker)
        mgr._qual_agent.answer.return_value = _make_qual_result()

        mocker.patch(
            "agents.manager.lc.invoke",
            return_value=_FakeLLMResponse(text=_class_json(route="qualitative")),
        )
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", return_value="test-qid-5")

        mgr.answer("First question?")
        assert len(mgr._history) == 1

        mocker.patch(
            "agents.manager.lc.invoke",
            return_value=_FakeLLMResponse(text=_class_json(route="qualitative")),
        )
        mocker.patch("agents.manager.new_query_id", return_value="test-qid-6")
        mgr.answer("Follow-up question?")
        assert len(mgr._history) == 2

    def test_history_passed_to_agents(self, mocker):
        """History messages must be forwarded to the sub-agents."""
        mgr = _make_manager(mocker)
        mgr._qual_agent.answer.return_value = _make_qual_result()

        mocker.patch(
            "agents.manager.lc.invoke",
            return_value=_FakeLLMResponse(text=_class_json(route="qualitative")),
        )
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", return_value="test-qid-7")

        # Seed one turn manually into history
        mgr._history.add(Turn("Q1", "A1", "prev-qid", "qualitative"))

        mgr.answer("Second question?")

        call_kwargs = mgr._qual_agent.answer.call_args
        history_arg = call_kwargs.kwargs.get("history") or call_kwargs.args[2]
        assert len(history_arg) == 2  # 1 turn = 2 messages
        assert history_arg[0]["content"] == "Q1"

    def test_reset_clears_manager_history(self, mocker):
        """manager.reset() must clear the session history."""
        mgr = _make_manager(mocker)
        mgr._history.add(Turn("Q", "A", "qid", "qualitative"))
        assert len(mgr._history) == 1
        mgr.reset()
        assert len(mgr._history) == 0


# ── Manager: error handling ───────────────────────────────────────────────────

class TestManagerErrorHandling:

    def test_classification_parse_error_returns_refused(self, mocker):
        """Malformed classification JSON → refused result without calling agents."""
        mgr = _make_manager(mocker)

        mocker.patch(
            "agents.manager.lc.invoke",
            return_value=_FakeLLMResponse(text="not json }{"),
        )
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", return_value="test-qid-8")

        result = mgr.answer("Some query")

        assert result.refused
        assert not result.validation.valid
        mgr._qual_agent.answer.assert_not_called()
        mgr._quant_agent.answer.assert_not_called()

    def test_invalid_route_fails_validation(self, mocker):
        """Unknown route value fails classification validation."""
        mgr = _make_manager(mocker)

        bad_class = json.dumps({
            "route": "magic_route",
            "qualitative_subquery": None,
            "quantitative_subquery": None,
            "reasoning": "test",
        })
        mocker.patch(
            "agents.manager.lc.invoke",
            return_value=_FakeLLMResponse(text=bad_class),
        )
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", return_value="test-qid-9")

        result = mgr.answer("Some query")
        assert result.refused
        assert any("route" in e for e in result.validation.errors)

    def test_query_id_present_in_result(self, mocker):
        """Every result must carry a query_id for tokenomics tracing."""
        mgr = _make_manager(mocker)
        mgr._qual_agent.answer.return_value = _make_qual_result()

        mocker.patch(
            "agents.manager.lc.invoke",
            return_value=_FakeLLMResponse(text=_class_json(route="qualitative")),
        )
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", return_value="abc123")

        result = mgr.answer("Q?")
        assert result.query_id == "abc123"

    def test_log_query_summary_called_exactly_once(self, mocker):
        """log_query_summary must be called exactly once per Manager.answer() call."""
        mgr = _make_manager(mocker)
        mgr._qual_agent.answer.return_value = _make_qual_result()

        mocker.patch(
            "agents.manager.lc.invoke",
            return_value=_FakeLLMResponse(text=_class_json(route="qualitative")),
        )
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mock_summary = mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", return_value="test-qid-10")

        mgr.answer("Q?")
        assert mock_summary.call_count == 1


# ── TestSynthesisValidation ───────────────────────────────────────────────────

class TestSynthesisValidation:
    """Validate validate_synthesis_response grounding behaviour."""

    def test_synthesis_prose_number_warns_not_errors(self):
        """Fabricated number not in quant rows or policy texts → hard error."""
        from validation.validator import validate_synthesis_response
        quant_rows = [{"count": 37.0}]
        payload = {
            "synthesis": (
                "The data shows 37 active employees. Also, 99999 is fabricated."
            ),
            "refused": False,
        }
        result = validate_synthesis_response(payload, "end_turn", quant_result_rows=quant_rows)
        # After fix: 99999 not in quant rows, no policy texts → hard error
        assert not result.valid, (
            f"Expected invalid — fabricated 99999 must be an error, got: {result}"
        )
        assert any("99999" in e or "not found" in e.lower() for e in result.errors), (
            f"Expected error mentioning 99999, got: {result.errors}"
        )

    def test_synthesis_number_in_quant_rows_passes(self):
        """Number in synthesis that matches quant rows → no error."""
        from validation.validator import validate_synthesis_response
        quant_rows = [{"revenue": 502000.0}]
        payload = {
            "synthesis": "The West region generated $502,000 in revenue.",
            "refused": False,
        }
        result = validate_synthesis_response(payload, "end_turn", quant_result_rows=quant_rows)
        assert result.valid, f"502000 is in quant rows — should pass: {result.errors}"


class TestSynthesisGrounding:
    """Regression tests for synthesis grounding fixes."""

    def test_invented_revenue_rejected(self):
        """Synthesis containing revenue not in quant rows must fail."""
        from validation.validator import validate_synthesis_response
        payload = {
            "synthesis": "Total revenue was $99,999 this quarter.",
            "refused": False,
        }
        result = validate_synthesis_response(
            payload, "end_turn",
            quant_result_rows=[{"total_revenue": 50000.0}],
        )
        assert not result.valid
        assert any("99999" in e or "not found" in e.lower() for e in result.errors)

    def test_invented_small_count_rejected(self):
        """Synthesis containing a small count not in quant rows must fail."""
        from validation.validator import validate_synthesis_response
        payload = {
            "synthesis": "Only 3 customers have churned.",
            "refused": False,
        }
        result = validate_synthesis_response(
            payload, "end_turn",
            quant_result_rows=[{"churned_customers": 8}],
            retrieved_policy_texts=[],
        )
        assert not result.valid
        assert any("3" in e or "not found" in e.lower() for e in result.errors)

    def test_supported_database_number_passes(self):
        """A number matching a quant result row must pass."""
        from validation.validator import validate_synthesis_response
        payload = {"synthesis": "8 customers have churned.", "refused": False}
        result = validate_synthesis_response(
            payload, "end_turn",
            quant_result_rows=[{"churned_customers": 8}],
        )
        assert result.valid, f"8 is in result rows — should pass: {result.errors}"

    def test_policy_grounded_number_passes(self):
        """A number that appears in retrieved policy text must pass."""
        from validation.validator import validate_synthesis_response
        payload = {
            "synthesis": "The policy requires passwords of at least 16 characters.",
            "refused": False,
        }
        result = validate_synthesis_response(
            payload, "end_turn",
            quant_result_rows=[],
            retrieved_policy_texts=["Passwords must be at least 16 characters."],
        )
        assert result.valid, f"16 is in policy text — should pass: {result.errors}"

    def test_manager_retries_synthesis_on_grounding_failure(self, mocker):
        """Manager makes a second synthesis call when the first fails grounding."""
        mgr = _make_manager(mocker)
        qual = _make_qual_result()
        qual.retrieved_chunks = [{"document": "Policy text with no numbers."}]
        quant = _make_quant_result()
        quant.result_rows = [{"total": 5000.0}]
        mgr._qual_agent.answer.return_value = qual
        mgr._quant_agent.answer.return_value = quant

        mock_invoke = mocker.patch(
            "agents.manager.lc.invoke",
            side_effect=[
                # classification → both
                _FakeLLMResponse(text=_class_json(
                    route="both",
                    qual_sub="Policy?",
                    quant_sub="Data?",
                )),
                # synthesis attempt 1: fabricated number not in result_rows
                _FakeLLMResponse(text='{"synthesis": "Revenue was $99999.", "refused": false, "refusal_reason": null}'),
                # synthesis attempt 2: grounded number
                _FakeLLMResponse(text='{"synthesis": "Revenue was $5000.", "refused": false, "refusal_reason": null}'),
            ],
        )
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", return_value="synth-retry-1")

        result = mgr.answer("Combined question?")

        # lc.invoke called 3 times: 1 classification + 2 synthesis
        assert mock_invoke.call_count == 3, (
            f"Expected 3 invoke calls (1 classify + 2 synthesis), got {mock_invoke.call_count}"
        )
        assert result.validation.valid, f"Second synthesis should pass: {result.validation.errors}"

    def test_fallback_used_and_stored_when_all_synthesis_retries_fail(self, mocker):
        """When all synthesis retries fail, fallback is used and stored in history."""
        mgr = _make_manager(mocker)
        qual = _make_qual_result()
        qual.retrieved_chunks = [{"document": "Policy has no relevant numbers."}]
        quant = _make_quant_result()
        quant.result_rows = [{"total": 5000.0}]
        mgr._qual_agent.answer.return_value = qual
        mgr._quant_agent.answer.return_value = quant

        mocker.patch(
            "agents.manager.lc.invoke",
            side_effect=[
                _FakeLLMResponse(text=_class_json(
                    route="both",
                    qual_sub="Policy?",
                    quant_sub="Data?",
                )),
                # synthesis attempt 1: fabricated
                _FakeLLMResponse(text='{"synthesis": "Revenue was $99999.", "refused": false, "refusal_reason": null}'),
                # synthesis attempt 2: still fabricated
                _FakeLLMResponse(text='{"synthesis": "Revenue was $77777.", "refused": false, "refusal_reason": null}'),
            ],
        )
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", return_value="synth-fallback-1")

        result = mgr.answer("Combined question?")

        # fallback used — final_answer must not contain the fabricated synthesis
        assert "99999" not in result.final_answer
        assert "77777" not in result.final_answer
        # fallback IS stored in history (user gets the concatenated answer)
        assert len(mgr._history) == 1, (
            f"Fallback answer should be stored in history, got {len(mgr._history)} turns"
        )


# ── Issue 1 regression tests ──────────────────────────────────────────────────

class TestRefusalPropagation:

    def test_qualitative_refusal_propagates_to_manager_result(self, mocker):
        """Manager.refused must be True when the routed qualitative agent refuses."""
        mgr = _make_manager(mocker)
        refused_qual = _make_qual_result(refused=True)
        mgr._qual_agent.answer.return_value = refused_qual

        mocker.patch("agents.manager.lc.invoke",
                     return_value=_FakeLLMResponse(text=_class_json(route="qualitative")))
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", return_value="ref-qid-1")

        result = mgr.answer("Some policy question?")

        assert result.refused is True
        assert result.refusal_reason is not None
        assert "No context" in result.refusal_reason

    def test_refused_answer_not_added_to_history(self, mocker):
        """A refused qualitative answer must not be stored in SessionHistory."""
        mgr = _make_manager(mocker)
        refused_qual = _make_qual_result(refused=True)
        good_qual = _make_qual_result(answer="Real policy answer.")

        mocker.patch("agents.manager.lc.invoke",
                     return_value=_FakeLLMResponse(text=_class_json(route="qualitative")))
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", side_effect=["ref-qid-2", "ref-qid-3"])

        # First call: refused
        mgr._qual_agent.answer.return_value = refused_qual
        mgr.answer("Refused question?")
        assert len(mgr._history) == 0, "Refused turn should not be in history"

        # Second call: valid
        mgr._qual_agent.answer.return_value = good_qual
        mocker.patch("agents.manager.lc.invoke",
                     return_value=_FakeLLMResponse(text=_class_json(route="qualitative")))
        mgr.answer("Good question?")
        assert len(mgr._history) == 1, "Valid turn should be in history"
        msgs = mgr._history.to_messages()
        assert "Real policy answer." in msgs[1]["content"]


class TestSynthesisRefusalPropagation:

    def test_synthesis_refused_true_uses_fallback(self, mocker):
        """synthesis refused=True must trigger fallback, not report success with empty answer."""
        mgr = _make_manager(mocker)
        mgr._qual_agent.answer.return_value = _make_qual_result()
        mgr._quant_agent.answer.return_value = _make_quant_result()

        mocker.patch(
            "agents.manager.lc.invoke",
            side_effect=[
                _FakeLLMResponse(text=_class_json(route="both", qual_sub="Policy?", quant_sub="Data?")),
                # synthesis attempt 1: refused=True, empty synthesis
                _FakeLLMResponse(text=json.dumps({
                    "synthesis": "", "refused": True, "refusal_reason": "Cannot combine."
                })),
                # synthesis attempt 2: also refused
                _FakeLLMResponse(text=json.dumps({
                    "synthesis": "", "refused": True, "refusal_reason": "Still cannot combine."
                })),
            ],
        )
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", return_value="synth-refused-1")

        result = mgr.answer("Combined question?")

        # Overall turn is NOT refused — agents produced valid answers
        assert not result.refused, "Manager should not report refused when agents succeeded"
        # final_answer must be non-empty (fallback)
        assert result.final_answer.strip(), "final_answer must not be empty after synthesis refusal"
        # final_answer must contain agent content, not be the raw refused synthesis
        assert "Policy answer" in result.final_answer or "Policy:" in result.final_answer

    def test_synthesis_refused_fallback_stored_in_history(self, mocker):
        """When synthesis refuses on all attempts, fallback is stored in history (not empty)."""
        mgr = _make_manager(mocker)
        mgr._qual_agent.answer.return_value = _make_qual_result()
        mgr._quant_agent.answer.return_value = _make_quant_result()

        mocker.patch(
            "agents.manager.lc.invoke",
            side_effect=[
                _FakeLLMResponse(text=_class_json(route="both", qual_sub="Policy?", quant_sub="Data?")),
                _FakeLLMResponse(text=json.dumps({
                    "synthesis": "", "refused": True, "refusal_reason": "Cannot combine."
                })),
                _FakeLLMResponse(text=json.dumps({
                    "synthesis": "", "refused": True, "refusal_reason": "Still cannot."
                })),
            ],
        )
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", return_value="synth-refused-2")

        result = mgr.answer("Combined question?")

        # History must have exactly one turn (the fallback)
        assert len(mgr._history) == 1, "Fallback answer should be stored in history"
        stored = mgr._history.to_messages()
        # The stored answer must be non-empty
        assistant_msgs = [m["content"] for m in stored if m["role"] == "assistant"]
        assert assistant_msgs and assistant_msgs[0].strip(), "Stored history must not be empty"


class TestSynthesisHistoryGuard:

    def test_invalid_synthesis_not_added_to_history(self, mocker):
        """
        When all synthesis retries fail, the fallback answer is used.
        The invalid synthesis text must not be stored in history — the
        fallback (concatenated) answer is stored instead.
        """
        mgr = _make_manager(mocker)
        mgr._qual_agent.answer.return_value = _make_qual_result()
        mgr._quant_agent.answer.return_value = _make_quant_result()

        mocker.patch(
            "agents.manager.lc.invoke",
            side_effect=[
                # classification → both
                _FakeLLMResponse(text=_class_json(
                    route="both",
                    qual_sub="Policy question",
                    quant_sub="Data question",
                )),
                # synthesis attempt 1 — truncated (fails validation)
                _FakeLLMResponse(
                    text='{"synthesis": "Combined.", "refused": false, "refusal_reason": null}',
                    stop_reason="max_tokens",
                ),
                # synthesis attempt 2 — also truncated (fails validation)
                _FakeLLMResponse(
                    text='{"synthesis": "Combined again.", "refused": false, "refusal_reason": null}',
                    stop_reason="max_tokens",
                ),
            ],
        )
        mocker.patch("agents.manager.log_call", return_value=MagicMock())
        mocker.patch("agents.manager.log_query_summary")
        mocker.patch("agents.manager.new_query_id", return_value="synth-guard-1")

        result = mgr.answer("Combined question?")

        # Synthesis validation is invalid (truncated)
        assert not result.validation.valid
        # Fallback answer used — not the synthesis text
        assert "Policy:" in result.final_answer or "Policy answer" in result.final_answer
        # Fallback IS added to history (composed of validated agent answers)
        assert len(mgr._history) == 1
