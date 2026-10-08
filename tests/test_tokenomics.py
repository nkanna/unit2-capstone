"""Real JSONL accounting in temporary directories; no model calls."""
import json
import pytest
from llm_client import LLMResponse,DEFAULT_MODEL,QUOTA_FALLBACK_MODEL
from tokenomics import logger

@pytest.fixture(autouse=True)
def isolated(monkeypatch,tmp_path):
    monkeypatch.setattr(logger,'LOG_PATH',tmp_path/'gemini.jsonl')
    monkeypatch.setattr(logger,'_now_iso',lambda:'2026-10-08T00:00:00+00:00')
    monkeypatch.delenv('GEMINI_BILLING_TIER',raising=False)

def response(model=DEFAULT_MODEL, input_tokens=1000, output_tokens=100):
    return LLMResponse('ok',input_tokens,output_tokens,'end_turn',model=model,
                       provider_finish_reason='stop',total_tokens=1100,
                       usage_metadata={'prompt_tokens':input_tokens,'completion_tokens':output_tokens,'total_tokens':1100})

def test_free_and_hypothetical_paid_are_separate(monkeypatch):
    monkeypatch.setenv('GEMINI_BILLING_TIER','free')
    r=logger.log_call('q','classification',response(),'pass')
    assert r.estimated_cost_usd==0.0 and r.free_tier_estimated_cost_usd==0.0
    assert r.hypothetical_paid_tier_cost_usd==0.001125
    assert r.provider=='google_gemini' and r.provider_finish_reason=='stop'

def test_unknown_billing_is_not_claimed_free():
    r=logger.log_call('q','test',response(),'pass')
    assert r.estimated_cost_usd is None and r.billing_tier=='unknown'
    assert r.hypothetical_paid_tier_cost_usd==0.001125

def test_fallback_uses_own_model_rate():
    r=logger.log_call('q','test',response(QUOTA_FALLBACK_MODEL),'pass')
    assert r.model==QUOTA_FALLBACK_MODEL and r.hypothetical_paid_tier_cost_usd==0.00055

def test_legacy_claude_rate_variables_are_ignored(monkeypatch):
    monkeypatch.setenv('COST_PER_M_INPUT','99'); monkeypatch.setenv('COST_PER_M_OUTPUT','99')
    r=logger.log_call('q','test',response(),'pass')
    assert r.hypothetical_paid_tier_cost_usd==0.001125

def test_cached_input_uses_cache_rate():
    resp=response(); resp.usage_metadata['prompt_tokens_details']={'cached_tokens':500}
    r=logger.log_call('q','test',resp,'pass')
    assert r.hypothetical_paid_tier_cost_usd==0.0007875

def test_query_totals_use_calls_once():
    calls=[logger.log_call('q','first',response(),'pass'),logger.log_call('q','retry',response(),'fail')]
    s=logger.log_query_summary('q','question',calls)
    rows=[json.loads(x) for x in logger.LOG_PATH.read_text().splitlines()]
    assert [r['record_type'] for r in rows]==['call','call','query_summary']
    assert s['call_count']==2 and s['total_input_tokens']==2000 and s['total_output_tokens']==200
    assert s['total_tokens']==2200 and s['total_hypothetical_paid_tier_cost_usd']==0.00225
    assert s['total_estimated_cost_usd'] is None

def test_unknown_usage_propagates_without_fabricated_tokens():
    r=logger.log_call('q','test',response(input_tokens=None),'fail')
    s=logger.log_query_summary('q','question',[r])
    assert r.input_tokens is None and r.hypothetical_paid_tier_cost_usd is None
    assert s['total_input_tokens'] is None and not s['usage_complete']

def test_unknown_model_has_unknown_price():
    r=logger.log_call('q','test',response('unpriced-model'),'pass')
    assert r.hypothetical_paid_tier_cost_usd is None and r.free_tier_estimated_cost_usd is None

def test_paid_tier_selects_paid_estimate(monkeypatch):
    monkeypatch.setenv('GEMINI_BILLING_TIER','paid')
    r=logger.log_call('q','test',response(),'pass')
    assert r.estimated_cost_usd==r.hypothetical_paid_tier_cost_usd

def test_historical_log_paths_are_blocked(monkeypatch):
    monkeypatch.setattr(logger,'LOG_PATH',logger.ROOT/'tokenomics_log.jsonl')
    with pytest.raises(ValueError,match='read-only'): logger.log_call('q','test',response(),'pass')

def test_paid_pricing_promotion_expiry():
    assert logger._rates(DEFAULT_MODEL,'2026-12-31')['input']==0.75
    assert logger._rates(DEFAULT_MODEL,'2027-01-01')['input']==1.50

def test_model_override_cannot_mislabel_real_usage():
    with pytest.raises(ValueError,match='differs'):
        logger.log_call('q','test',response(),'pass',model=QUOTA_FALLBACK_MODEL)

def test_compatibility_thinking_tokens_are_derived_and_billed_once(monkeypatch):
    """Reproduce the usage shape from the author's successful live smoke call."""
    monkeypatch.setenv('GEMINI_BILLING_TIER', 'paid')
    resp = LLMResponse('Hello!', 8, 10, 'end_turn', total_tokens=257,
                       usage_metadata={'prompt_tokens': 8, 'completion_tokens': 10,
                                       'total_tokens': 257,
                                       'completion_tokens_details': None})
    record = logger.log_call('q', 'smoke_test', resp, 'pass')
    assert (record.input_tokens, record.output_tokens, record.total_tokens) == (8, 10, 257)
    assert record.usage_metadata == resp.usage_metadata
    assert record.thinking_tokens == 239
    assert record.thinking_tokens_source == 'derived_from_provider_totals'
    assert record.billable_output_tokens == 249
    assert record.hypothetical_paid_tier_cost_usd == round((8 * 0.75 + 249 * 3.75) / 1_000_000, 8)
    summary = logger.log_query_summary('q', 'Say hello.', [record])
    assert summary['total_output_tokens'] == 10
    assert summary['total_thinking_tokens'] == 239
    assert summary['total_billable_output_tokens'] == 249
    assert summary['total_tokens'] == 257
    assert summary['usage_accounting_complete']

def test_reasoning_detail_is_not_added_twice():
    resp = response()
    resp.total_tokens = 1300
    resp.usage_metadata.update(total_tokens=1300, completion_tokens_details={'reasoning_tokens': 200})
    record = logger.log_call('q', 'test', resp, 'pass')
    assert record.thinking_tokens == 200 and record.billable_output_tokens == 300
    assert record.hypothetical_paid_tier_cost_usd == 0.001875

@pytest.mark.parametrize('total', [None, 1099])
def test_unknown_or_inconsistent_totals_do_not_invent_thinking_or_paid_cost(total):
    resp = response()
    resp.total_tokens = total
    record = logger.log_call('q', 'test', resp, 'pass')
    assert record.thinking_tokens is None and record.billable_output_tokens is None
    assert record.hypothetical_paid_tier_cost_usd is None
    summary = logger.log_query_summary('q', 'question', [record])
    assert summary['total_thinking_tokens'] is None
    assert not summary['usage_accounting_complete']
