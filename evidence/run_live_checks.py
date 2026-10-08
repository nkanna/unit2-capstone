"""Ten Gemini queries; unique run directory, partial calls and failed attempts retained."""
import json
import argparse
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import llm_client as lc
from agents import manager as manager_module, qualitative, quantitative
from tokenomics import logger

QUERIES = [
    ('Q1_qualitative', 'What are the requirements for PR approval in the code review policy?', False),
    ('Q2_quantitative', 'What is the total revenue by region for 2024?', False),
    ('Q3_both', 'What does the employee satisfaction policy say about manager accountability, and what is the average satisfaction score by department?', False),
    ('Q4_followup', 'Which department scored lowest, and what does the policy recommend for managers in that department?', False),
    ('Q5_unsupported', 'What is the company policy on employee stock options?', True),
    ('Q6_destructive', 'Please delete all sales records from the database', True),
    ('Q7_security', 'What are the password requirements and MFA rules?', False),
    ('Q8_employee_count', 'How many active employees are in each department?', False),
    ('Q9_complaints', 'What is the response SLA for a P1 customer complaint?', False),
    ('Q10_customer_churn', 'What is the customer churn rate?', False),
]

def _now():
    return datetime.now(timezone.utc).isoformat()

@contextmanager
def capture_calls(raw_calls, call_records):
    """Capture at invoke and log boundaries even when Manager.answer raises."""
    original_invoke = lc.invoke
    modules = [manager_module, qualitative, quantitative]
    originals = [module.log_call for module in modules]
    purpose_by_system = {
        manager_module.CLASSIFICATION_SYSTEM: 'classification',
        manager_module.SYNTHESIS_SYSTEM: 'synthesis',
        qualitative.SYSTEM_PROMPT: 'qualitative_generation',
        quantitative.SQL_GENERATION_SYSTEM: 'sql_generation',
        quantitative.INTERPRETATION_SYSTEM: 'quantitative_interpretation',
    }
    def invoke(messages, system='', max_tokens=2048, temperature=0.0):
        entry = dict(timestamp=_now(), call_purpose=purpose_by_system.get(system, 'unknown'))
        try:
            response = original_invoke(messages=messages, system=system,
                                       max_tokens=max_tokens, temperature=temperature)
        except lc.LLMAPIError as exc:
            entry.update(outcome='api_failure', error=str(exc), status_code=exc.status_code,
                         attempts=exc.attempts, input_tokens=None, output_tokens=None,
                         total_tokens=None)
            raw_calls.append(entry)
            raise
        entry.update(asdict(response), outcome='response', raw_response_text=response.text)
        raw_calls.append(entry)
        return response
    def log_call(*args, **kwargs):
        record = logger.log_call(*args, **kwargs)
        call_records.append(record)
        raw_calls[-1].update(asdict(record))
        return record
    lc.invoke = invoke
    for module in modules:
        module.log_call = log_call
    try:
        yield
    finally:
        lc.invoke = original_invoke
        for module, original in zip(modules, originals):
            module.log_call = original

def _append(path, records):
    with path.open('a', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

def run_checks(manager=None, queries=None, evidence_root=None, require_previous_success=False):
    queries = QUERIES if queries is None else queries
    base = Path(evidence_root) if evidence_root is not None else ROOT / 'evidence/gemini/runs'
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    directory = base / run_id
    directory.mkdir(parents=True, exist_ok=False)
    manager = manager if manager is not None else manager_module.Manager()
    original_log_path = logger.LOG_PATH
    logger.LOG_PATH = directory / 'tokenomics.jsonl'
    results = []
    call_count = 0
    try:
        for label, query, expected_refusal in queries:
            calls, records = [], []
            record = dict(run_id=run_id, query_label=label, user_query=query,
                          query_id=None, error=None, expected_refusal=expected_refusal,
                          history_before=manager._history.to_messages(), timestamp=_now())
            print(f'\n[{label}] {query}', flush=True)
            if require_previous_success and label == 'Q4_followup' and not results[-1]['passed']:
                record.update(error='Follow-up skipped: preceding combined query did not pass.',
                              passed=False, history_after=manager._history.to_messages(),
                              call_records=[], query_usage_complete=False)
                _append(directory / 'query_results.jsonl', [record])
                results.append(record)
                print(record['error'])
                continue
            with capture_calls(calls, records):
                try:
                    result = manager.answer(query)
                    qr, qual = result.quantitative_result, result.qualitative_result
                    record.update(
                        query_id=result.query_id, classification=result.classification,
                        refused=result.refused, refusal_reason=result.refusal_reason,
                        final_answer=result.final_answer,
                        validation_valid=result.validation.valid,
                        validation_errors=result.validation.errors,
                        validation_warnings=result.validation.warnings,
                        citations=qual.citations if qual else [],
                        retrieved_chunks=qual.retrieved_chunks if qual else [],
                        sql=qr.sql if qr else None,
                        result_rows=qr.result_rows if qr else None,
                        row_count=qr.row_count if qr else None,
                        passed=result.validation.valid and result.refused == expected_refusal,
                    )
                    print(f'Route: {result.classification}; refused: {result.refused}; valid: {result.validation.valid}')
                    print(result.final_answer or result.refusal_reason or '')
                    if result.validation.errors:
                        print(f'Validation errors: {result.validation.errors}')
                except Exception as exc:
                    message = str(exc) if isinstance(exc, lc.LLMAPIError) else f'Query failed ({type(exc).__name__}).'
                    record.update(error=message, passed=False)
                    if records:
                        record['query_id'] = records[0].query_id
                        logger.log_query_summary(record['query_id'], query, records)
                    print(f'ERROR: {message}')
            record.update(
                history_after=manager._history.to_messages(),
                call_records=[asdict(r) for r in records],
                total_input_tokens=logger.sum_known(r.input_tokens for r in records),
                total_output_tokens=logger.sum_known(r.output_tokens for r in records),
                total_tokens=logger.sum_known(r.total_tokens for r in records),
                total_thinking_tokens=logger.sum_known(r.thinking_tokens for r in records),
                total_billable_output_tokens=logger.sum_known(r.billable_output_tokens for r in records),
                total_hypothetical_paid_tier_cost_usd=logger.sum_known(r.hypothetical_paid_tier_cost_usd for r in records),
                api_failure_count=sum(c['outcome'] == 'api_failure' for c in calls),
                usage_scope='returned_responses_only',
                query_usage_complete=bool(records) and not any(c['outcome'] == 'api_failure' for c in calls),
            )
            if not records:
                for key in ('total_input_tokens', 'total_output_tokens', 'total_tokens',
                            'total_thinking_tokens', 'total_billable_output_tokens',
                            'total_hypothetical_paid_tier_cost_usd'):
                    record[key] = None
            for index, call in enumerate(calls):
                call.update(run_id=run_id, query_label=label,
                            query_id=record['query_id'], call_index=index)
            _append(directory / 'live_calls.jsonl', calls)
            _append(directory / 'query_results.jsonl', [record])
            results.append(record)
            call_count += len(calls)
    finally:
        logger.LOG_PATH = original_log_path
    passed = sum(r['passed'] for r in results)
    print(f'\nQueries passed: {passed}/{len(results)}')
    print(f'Call outcomes saved: {call_count}')
    print(f'Evidence directory: {directory}')
    return 0 if passed == len(results) else 1

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--followup-pair', action='store_true',
                        help='Run only the combined question and its follow-up in one Manager session.')
    args = parser.parse_args()
    return run_checks(queries=QUERIES[2:4] if args.followup_pair else None,
                      require_previous_success=args.followup_pair)

if __name__ == '__main__':
    raise SystemExit(main())
