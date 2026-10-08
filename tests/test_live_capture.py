"""Evidence failure recovery without real credentials or retrieval downloads."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from evidence import run_live_checks as capture
from llm_client import LLMResponse, LLMAPIError
from validation.validator import ValidationResult


class FakeManager:
    def __init__(self, fail=False):
        self._history = SimpleNamespace(to_messages=lambda: [])
        self.fail = fail

    def answer(self, query):
        response = capture.lc.invoke([{'role': 'user', 'content': query}],
                                     system=capture.manager_module.CLASSIFICATION_SYSTEM)
        call = capture.manager_module.log_call('offline-q', 'classification', response, 'pass')
        if self.fail:
            capture.lc.invoke([{'role': 'user', 'content': query}],
                              system=capture.qualitative.SYSTEM_PROMPT)
        return capture.manager_module.ManagerResult(
            'Outside supported scope.', 'unsupported', True, 'unsupported',
            None, None, ValidationResult(valid=True), 'offline-q', [call])


def response():
    return LLMResponse('ok', 5, 2, 'end_turn', total_tokens=7)


def test_failure_retains_partial_usage_and_api_attempts(monkeypatch, tmp_path):
    invoke = MagicMock(side_effect=[response(), LLMAPIError('Gemini request failed (HTTP 503).', 503,
                      [{'model': 'gemini-3.6-flash', 'status_code': 503, 'outcome': 'api_failure'}])])
    monkeypatch.setattr(capture.lc, 'invoke', invoke)
    old_log = capture.logger.LOG_PATH
    assert capture.run_checks(FakeManager(fail=True), [('Q1', 'Policy?', False)], tmp_path) == 1
    directory = next(tmp_path.iterdir())
    calls = [json.loads(line) for line in (directory / 'live_calls.jsonl').read_text().splitlines()]
    assert len(calls) == 2
    assert calls[0]['query_id'] == calls[1]['query_id'] == 'offline-q'
    assert calls[0]['call_purpose'] == 'classification' and calls[0]['input_tokens'] == 5
    assert calls[1]['call_purpose'] == 'qualitative_generation'
    assert calls[1]['input_tokens'] is None and calls[1]['attempts'][0]['status_code'] == 503
    result = json.loads((directory / 'query_results.jsonl').read_text())
    assert not result['passed'] and result['api_failure_count'] == 1
    assert result['total_tokens'] == 7
    ledger = [json.loads(line) for line in (directory / 'tokenomics.jsonl').read_text().splitlines()]
    assert [r['record_type'] for r in ledger] == ['call', 'query_summary']
    assert capture.logger.LOG_PATH == old_log and capture.lc.invoke is invoke


def test_new_run_keeps_prior_capture_and_expected_refusal_passes(monkeypatch, tmp_path):
    monkeypatch.setattr(capture.lc, 'invoke', lambda **_: response())
    query = [('Q6', 'Delete records.', True)]
    assert capture.run_checks(FakeManager(), query, tmp_path) == 0
    first = next(tmp_path.iterdir())
    original = (first / 'query_results.jsonl').read_bytes()
    assert capture.run_checks(FakeManager(), query, tmp_path) == 0
    assert len(list(tmp_path.iterdir())) == 2
    assert (first / 'query_results.jsonl').read_bytes() == original


def test_supported_query_refusal_does_not_count_as_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(capture.lc, 'invoke', lambda **_: response())
    assert capture.run_checks(FakeManager(), [('Q1', 'Policy?', False)], tmp_path) == 1


def test_ten_unique_queries_and_followup_order():
    assert len(capture.QUERIES) == 10
    assert len({q for _, q, _ in capture.QUERIES}) == 10
    assert [label for label, _, _ in capture.QUERIES][2:4] == ['Q3_both', 'Q4_followup']

def test_failed_combined_query_does_not_attempt_contextless_followup(monkeypatch, tmp_path):
    invoke = MagicMock(return_value=response())
    monkeypatch.setattr(capture.lc, 'invoke', invoke)
    assert capture.run_checks(FakeManager(), capture.QUERIES[2:4], tmp_path,
                              require_previous_success=True) == 1
    assert invoke.call_count == 1
    path = next(tmp_path.iterdir()) / 'query_results.jsonl'
    results = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(results) == 2 and not results[1]['passed']
    assert 'Follow-up skipped' in results[1]['error']
