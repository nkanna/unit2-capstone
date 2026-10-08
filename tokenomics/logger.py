"""Gemini usage and per-query accounting; historical Bedrock logs are read-only.
Rates: https://ai.google.dev/gemini-api/docs/pricing verified 2026-10-08.
Free estimates are conditional. The API does not identify account billing tier.
"""
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
PRICING_SOURCE = 'https://ai.google.dev/gemini-api/docs/pricing'
PRICING_VERIFIED = '2026-10-08'
PRICING_ASSUMPTION = 'Google Gemini Developer API Standard; estimates only, not measured charges'
TOKEN_ACCOUNTING_SOURCE = 'https://discuss.ai.google.dev/t/gemini-3-6-flash-openai-compatible-token-limits-response-fields-and-auth-keys/183372'
ROOT = Path(__file__).resolve().parent.parent
_path = os.getenv('GEMINI_TOKENOMICS_LOG', '').strip()
LOG_PATH = Path(_path if _path and not _path.startswith('<') else 'tokenomics_gemini_log.jsonl')

@dataclass
class CallRecord:
    record_type: str
    query_id: str
    call_purpose: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    stop_reason: str
    validation_outcome: str
    estimated_cost_usd: float | None
    pricing_assumption: str
    timestamp: str
    provider: str
    provider_finish_reason: str | None
    total_tokens: int | None
    usage_metadata: dict
    response_issue: str | None
    free_tier_estimated_cost_usd: float | None
    hypothetical_paid_tier_cost_usd: float | None
    billing_tier: str
    pricing_source: str
    pricing_verified_date: str
    pricing_rates_per_million: dict | None
    requested_model: str
    model_reported: bool
    attempts: list[dict]
    thinking_tokens: int | None
    thinking_tokens_source: str
    billable_output_tokens: int | None
    usage_accounting_issue: str | None

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _rates(model: str, date: str) -> dict | None:
    if model == 'gemini-3.6-flash':
        # Google's current promotional Standard rates expire Dec 31, 2026.
        return dict(input=0.75, output=3.75, cached_input=0.075) if date < '2027-01-01' else dict(input=1.50, output=7.50, cached_input=0.15)
    if model == 'gemini-3.5-flash-lite':
        return dict(input=0.30, output=2.50, cached_input=0.03)
    return None

def _output_accounting(response) -> dict:
    """For this text-only Gemini compatibility client, completion excludes
    thinking, while total includes it. Preserve provider values; identify the
    breakdown as arithmetic derived from returned totals, not an explicit
    provider thoughts-token field. No counts are filled in when missing.
    """
    counts = [response.input_tokens, response.output_tokens, response.total_tokens]
    issue = None
    if any(v is None for v in counts):
        issue = 'missing_usage'
    elif any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts):
        issue = 'invalid_usage'
    elif response.total_tokens < response.input_tokens + response.output_tokens:
        issue = 'inconsistent_usage_totals'
    if issue:
        return dict(thinking_tokens=None, thinking_tokens_source='unknown',
                    billable_output_tokens=None, usage_accounting_issue=issue)
    return dict(
        thinking_tokens=response.total_tokens - response.input_tokens - response.output_tokens,
        thinking_tokens_source='derived_from_provider_totals',
        billable_output_tokens=response.total_tokens - response.input_tokens,
        usage_accounting_issue=None,
    )

def _paid_estimate(response, rates: dict | None) -> float | None:
    billable_output = _output_accounting(response)['billable_output_tokens']
    if rates is None or billable_output is None:
        return None
    details = response.usage_metadata.get('prompt_tokens_details') or {}
    cached = details.get('cached_tokens', 0)
    if not isinstance(cached, int) or isinstance(cached, bool) or not 0 <= cached <= response.input_tokens:
        return None
    return round(((response.input_tokens - cached) * rates['input'] + cached * rates['cached_input'] + billable_output * rates['output']) / 1_000_000, 8)

def _append(record: dict) -> None:
    path = LOG_PATH.resolve()
    historical = [ROOT / 'tokenomics_log.jsonl', ROOT / 'evidence/live_calls.jsonl', ROOT / 'evidence/query_results.jsonl', ROOT / 'evidence/run_live_checks_v2/live_calls_v2.jsonl', ROOT / 'evidence/run_live_checks_v2/query_results_v2.jsonl']
    if path in [p.resolve() for p in historical]:
        raise ValueError('Historical Bedrock logs are read-only. Choose a separate Gemini log path.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

def log_call(query_id: str, call_purpose: str, response, validation_outcome: str, model: str | None = None) -> CallRecord:
    """Log returned usage, including invalid responses. 429s have no usage;
    attempt statuses are retained instead of invented billable call records.
    """
    if model is not None and model != response.model:
        raise ValueError('Model override differs from the provider response.')
    timestamp = _now_iso()
    rates = _rates(response.model, timestamp[:10])
    paid = _paid_estimate(response, rates)
    free = 0.0 if rates is not None else None
    tier = os.getenv('GEMINI_BILLING_TIER', 'unknown').strip().lower()
    if tier not in {'free', 'paid', 'unknown'}:
        tier = 'unknown'
    selected_cost = free if tier == 'free' else paid if tier == 'paid' else None
    record = CallRecord(
        record_type='call', query_id=query_id, call_purpose=call_purpose,
        model=response.model, input_tokens=response.input_tokens,
        output_tokens=response.output_tokens, stop_reason=response.stop_reason,
        validation_outcome=validation_outcome, estimated_cost_usd=selected_cost,
        pricing_assumption=PRICING_ASSUMPTION, timestamp=timestamp,
        provider=response.provider, provider_finish_reason=response.provider_finish_reason,
        total_tokens=response.total_tokens, usage_metadata=response.usage_metadata,
        response_issue=response.response_issue, free_tier_estimated_cost_usd=free,
        hypothetical_paid_tier_cost_usd=paid, billing_tier=tier,
        pricing_source=PRICING_SOURCE, pricing_verified_date=PRICING_VERIFIED,
        pricing_rates_per_million=rates, requested_model=response.requested_model,
        model_reported=response.model_reported, attempts=response.attempts,
        **_output_accounting(response),
    )
    _append(asdict(record))
    return record

def sum_known(values) -> int | float | None:
    values = list(values)
    return None if any(v is None for v in values) else sum(values)

def log_query_summary(query_id: str, user_query: str, call_records: list[CallRecord]) -> dict:
    """One summary per query; never sum calls and summaries together."""
    summary = {
        'record_type': 'query_summary', 'query_id': query_id, 'user_query': user_query,
        'provider': 'google_gemini', 'models': sorted({r.model for r in call_records}),
        'call_count': len(call_records), 'purposes': [r.call_purpose for r in call_records],
        'total_input_tokens': sum_known(r.input_tokens for r in call_records),
        'total_output_tokens': sum_known(r.output_tokens for r in call_records),
        'total_tokens': sum_known(r.total_tokens for r in call_records),
        'total_thinking_tokens': sum_known(r.thinking_tokens for r in call_records),
        'thinking_tokens_source': 'derived_from_provider_totals',
        'token_accounting_source': TOKEN_ACCOUNTING_SOURCE,
        'total_billable_output_tokens': sum_known(r.billable_output_tokens for r in call_records),
        'usage_accounting_complete': all(r.usage_accounting_issue is None for r in call_records),
        'total_estimated_cost_usd': sum_known(r.estimated_cost_usd for r in call_records),
        'total_free_tier_estimated_cost_usd': sum_known(r.free_tier_estimated_cost_usd for r in call_records),
        'total_hypothetical_paid_tier_cost_usd': sum_known(r.hypothetical_paid_tier_cost_usd for r in call_records),
        'usage_complete': all(r.input_tokens is not None and r.output_tokens is not None and r.total_tokens is not None for r in call_records),
        'billing_tiers': sorted({r.billing_tier for r in call_records}),
        'pricing_assumption': PRICING_ASSUMPTION, 'pricing_source': PRICING_SOURCE,
        'pricing_verified_date': PRICING_VERIFIED, 'timestamp': _now_iso(),
    }
    _append(summary)
    return summary

def new_query_id() -> str:
    return uuid.uuid4().hex[:12]
