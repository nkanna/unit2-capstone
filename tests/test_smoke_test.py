"""Smoke script behavior offline; no real credential or network dependency."""
import json
import pytest
import smoke_test
from llm_client import LLMResponse, LLMAPIError
from tokenomics import logger

@pytest.fixture(autouse=True)
def isolated(monkeypatch,tmp_path):
    monkeypatch.setattr(smoke_test,'EVIDENCE_PATH',tmp_path/'smoke.jsonl')
    monkeypatch.setattr(logger,'LOG_PATH',tmp_path/'tokens.jsonl')
    monkeypatch.delenv('GEMINI_BILLING_TIER',raising=False)

def test_success_requires_complete_response_and_actual_usage(monkeypatch,capsys):
    response=LLMResponse('Hello!',5,2,'end_turn',provider_finish_reason='stop',total_tokens=7,
                         usage_metadata={'prompt_tokens':5,'completion_tokens':2,'total_tokens':7})
    monkeypatch.setattr(smoke_test,'invoke',lambda **_:response)
    assert smoke_test.main()==0
    evidence=json.loads(smoke_test.EVIDENCE_PATH.read_text())
    assert evidence['status']=='pass' and evidence['response']['total_tokens']==7
    rows=[json.loads(x) for x in logger.LOG_PATH.read_text().splitlines()]
    assert rows[0]['record_type']=='call' and rows[1]['record_type']=='query_summary'
    assert rows[1]['total_tokens']==7

@pytest.mark.parametrize('reason,issue',[('max_tokens','truncated_response'),('safety_block','safety_block'),('missing_content','missing_content')])
def test_invalid_response_fails_smoke(monkeypatch,reason,issue,capsys):
    response=LLMResponse('partial',5,2,reason,total_tokens=7,response_issue=issue)
    monkeypatch.setattr(smoke_test,'invoke',lambda **_:response)
    assert smoke_test.main()==1
    assert json.loads(smoke_test.EVIDENCE_PATH.read_text())['status']=='fail'

def test_missing_usage_fails_without_invented_tokens(monkeypatch,capsys):
    response=LLMResponse('Hello!',None,None,'end_turn')
    monkeypatch.setattr(smoke_test,'invoke',lambda **_:response)
    assert smoke_test.main()==1
    rows=[json.loads(x) for x in logger.LOG_PATH.read_text().splitlines()]
    assert rows[0]['input_tokens'] is None and rows[1]['total_tokens'] is None

def test_api_failure_writes_failure_evidence_only(monkeypatch,capsys):
    def fail(**_): raise LLMAPIError('Gemini request failed (HTTP 429); 3 attempt(s).',429)
    monkeypatch.setattr(smoke_test,'invoke',fail)
    assert smoke_test.main()==1
    evidence=json.loads(smoke_test.EVIDENCE_PATH.read_text())
    assert evidence['status']=='fail' and evidence['status_code']==429
    assert not logger.LOG_PATH.exists()
