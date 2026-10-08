"""
Live evidence run v2 — combined + follow-up queries after code-fence fix.
Saves new Gemini evidence to evidence/gemini/run_live_checks_v2/.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import llm_client as lc
from agents.manager import Manager
from tokenomics.logger import sum_known

OUT_DIR = Path(__file__).parent.parent / "gemini" / "run_live_checks_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
_original_invoke = lc.invoke
_live_calls: list[dict] = []


def _capturing_invoke(messages, system, max_tokens, temperature=0.0):
    resp = _original_invoke(
        messages=messages, system=system,
        max_tokens=max_tokens, temperature=temperature,
    )
    _live_calls.append({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "text": resp.text,
        "stop_reason": resp.stop_reason,
        "provider_finish_reason": resp.provider_finish_reason,
        "provider": resp.provider,
        "model": resp.model,
        "usage_metadata": resp.usage_metadata,
        "total_tokens": resp.total_tokens,
        "response_issue": resp.response_issue,
        "attempts": resp.attempts,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
    })
    return resp


lc.invoke = _capturing_invoke

manager = Manager()

queries = [
    (
        "Q1_both",
        "What does the employee satisfaction policy say about manager accountability, "
        "and what is the average satisfaction score by department?",
    ),
    (
        "Q2_followup",
        "Which department scored lowest, and what does the policy recommend "
        "for managers in that department?",
    ),
]

query_results: list[dict] = []
call_start_idx = 0

for label, query in queries:
    print(f"\n{'=' * 70}")
    print(f"[{label}] {query}")
    history_before = manager._history.to_messages()

    try:
        result = manager.answer(query)
    except Exception as exc:
        print(f"ERROR: {exc}")
        query_results.append({"query_label": label, "user_query": query, "error": str(exc)})
        call_start_idx = len(_live_calls)
        continue

    history_after = manager._history.to_messages()
    query_calls = _live_calls[call_start_idx:]
    call_start_idx = len(_live_calls)

    for i, call in enumerate(query_calls):
        cr = result.call_records[i] if i < len(result.call_records) else None
        call["query_label"] = label
        call["query_id"] = result.query_id
        call["call_index"] = i
        call["call_purpose"] = (
            cr.call_purpose if cr and hasattr(cr, "call_purpose") else "unknown"
        )
        call["free_tier_estimated_cost_usd"] = cr.free_tier_estimated_cost_usd if cr else None
        call["hypothetical_paid_tier_cost_usd"] = cr.hypothetical_paid_tier_cost_usd if cr else None
        call["estimated_cost_usd"] = (
            cr.estimated_cost_usd if cr and hasattr(cr, "estimated_cost_usd") else None
        )

    qr: dict = {
        "query_label": label,
        "query_id": result.query_id,
        "user_query": query,
        "classification": result.classification,
        "refused": result.refused,
        "refusal_reason": result.refusal_reason,
        "final_answer": result.final_answer,
        "validation_valid": result.validation.valid,
        "validation_errors": result.validation.errors,
        "validation_warnings": result.validation.warnings,
        "citations": (
            result.qualitative_result.citations
            if result.qualitative_result and not result.qualitative_result.refused
            else []
        ),
        "sql": (
            result.quantitative_result.sql
            if result.quantitative_result and not result.quantitative_result.refused
            else None
        ),
        "result_rows": (
            result.quantitative_result.result_rows[:5]
            if result.quantitative_result and not result.quantitative_result.refused
            else None
        ),
        "row_count": (
            result.quantitative_result.row_count
            if result.quantitative_result and not result.quantitative_result.refused
            else None
        ),
        "history_before": history_before,
        "history_after": history_after,
        "total_input_tokens": sum_known(r.input_tokens for r in result.call_records),
        "total_output_tokens": sum_known(r.output_tokens for r in result.call_records),
        "total_hypothetical_paid_tier_cost_usd": sum_known(r.hypothetical_paid_tier_cost_usd for r in result.call_records),
        "total_free_tier_estimated_cost_usd": sum_known(r.free_tier_estimated_cost_usd for r in result.call_records),
        "total_cost_usd": sum_known(r.estimated_cost_usd for r in result.call_records),
    }
    query_results.append(qr)

    print(f"Route: {result.classification}")
    print(f"Refused: {result.refused}")
    if result.refused:
        print(f"Refusal: {result.refusal_reason}")
    else:
        print(f"Answer:\n{result.final_answer}")
    if result.qualitative_result and not result.qualitative_result.refused:
        cits = [
            (c["chunk_id"], c.get("excerpt", "")[:80])
            for c in result.qualitative_result.citations[:3]
        ]
        print(f"Citations (first 3): {cits}")
    if result.quantitative_result and not result.quantitative_result.refused:
        print(f"SQL: {result.quantitative_result.sql}")
        print(f"Rows: {result.quantitative_result.result_rows}")
    print(f"Validation: valid={result.validation.valid} "
          f"errors={result.validation.errors[:2]} "
          f"warnings={result.validation.warnings[:2]}")
    cost = sum_known(r.estimated_cost_usd for r in result.call_records)
    tokens_in = sum_known(r.input_tokens for r in result.call_records)
    tokens_out = sum_known(r.output_tokens for r in result.call_records)
    print(f"Tokens: {tokens_in} in / {tokens_out} out | Billing-tier estimate: {cost} USD")
    print(f"Query ID: {result.query_id}")

# Write evidence
calls_path = OUT_DIR / "live_calls_v2.jsonl"
results_path = OUT_DIR / "query_results_v2.jsonl"

with open(calls_path, "w") as f:
    for c in _live_calls:
        f.write(json.dumps(c) + "\n")

with open(results_path, "w") as f:
    for r in query_results:
        f.write(json.dumps(r) + "\n")

print(f"\nSaved {len(_live_calls)} calls → {calls_path}")
print(f"Saved {len(query_results)} results → {results_path}")
