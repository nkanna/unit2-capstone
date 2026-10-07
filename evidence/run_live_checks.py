"""
Live evidence capture for capstone submission.

Runs 6 queries through the complete Manager pipeline and saves:
  evidence/live_calls.jsonl  — one record per Bedrock call (raw text, tokens, cost)
  evidence/query_results.jsonl — one record per query (answer, citations, SQL, validation)

AWS credentials are never logged.  Run from the project root:
    python evidence/run_live_checks.py
"""

import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import bedrock_client as bc
from agents.manager import Manager

# ── Call interceptor ───────────────────────────────────────────────────────────

_original_invoke = bc.invoke
_captured_calls: list[dict] = []


def _capturing_invoke(messages, system, max_tokens, temperature=0.0):
    resp = _original_invoke(
        messages=messages, system=system,
        max_tokens=max_tokens, temperature=temperature,
    )
    _captured_calls.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": resp.text,
        "stop_reason": resp.stop_reason,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
    })
    return resp


bc.invoke = _capturing_invoke

# ── Queries ────────────────────────────────────────────────────────────────────

QUERIES = [
    ("Q1_qualitative",  "What are the requirements for PR approval in the code review policy?"),
    ("Q2_quantitative", "What is the total revenue by region for 2024?"),
    ("Q3_both",         "What does the employee satisfaction policy say about manager accountability, and what is the average satisfaction score by department?"),
    ("Q4_followup",     "Which department scored lowest, and what does the policy recommend for managers in that department?"),
    ("Q5_unsupported",  "What is the company policy on employee stock options?"),
    ("Q6_destructive",  "Please delete all sales records from the database"),
]

EVIDENCE_DIR = ROOT / "evidence"
CALLS_PATH   = EVIDENCE_DIR / "live_calls.jsonl"
RESULTS_PATH = EVIDENCE_DIR / "query_results.jsonl"

# ── Run ────────────────────────────────────────────────────────────────────────

manager = Manager()
all_query_records: list[dict] = []
all_call_records:  list[dict] = []

for label, query in QUERIES:
    call_count_before = len(_captured_calls)
    history_before = manager._history.to_messages()

    print(f"\n{'='*70}")
    print(f"[{label}] {query}")

    record: dict = {
        "query_label": label,
        "user_query": query,
        "error": None,
    }

    try:
        result = manager.answer(query)
        history_after = manager._history.to_messages()

        record.update({
            "query_id":          result.query_id,
            "classification":    result.classification,
            "refused":           result.refused,
            "refusal_reason":    result.refusal_reason,
            "final_answer":      result.final_answer,
            "validation_valid":  result.validation.valid,
            "validation_errors": result.validation.errors,
            "validation_warnings": result.validation.warnings,
            "citations": (
                [{"chunk_id": c.get("chunk_id"), "source": c.get("source"),
                  "excerpt": c.get("excerpt")}
                 for c in result.qualitative_result.citations]
                if result.qualitative_result and not result.qualitative_result.refused
                else []
            ),
            "sql": (
                result.quantitative_result.sql
                if result.quantitative_result and not result.quantitative_result.refused
                else None
            ),
            "result_rows": (
                result.quantitative_result.result_rows
                if result.quantitative_result and not result.quantitative_result.refused
                else None
            ),
            "row_count": (
                result.quantitative_result.row_count
                if result.quantitative_result and not result.quantitative_result.refused
                else None
            ),
            "history_before": history_before,
            "history_after":  history_after,
            "call_records": [
                {
                    "call_purpose":      cr.call_purpose,
                    "input_tokens":      cr.input_tokens,
                    "output_tokens":     cr.output_tokens,
                    "estimated_cost_usd": cr.estimated_cost_usd,
                    "validation_outcome": cr.validation_outcome,
                }
                for cr in result.call_records
            ],
            "attempts": (
                result.qualitative_result.attempts
                if result.qualitative_result
                else (result.quantitative_result.attempts if result.quantitative_result else None)
            ),
            "total_input_tokens":  sum(cr.input_tokens for cr in result.call_records),
            "total_output_tokens": sum(cr.output_tokens for cr in result.call_records),
            "total_cost_usd":      round(sum(cr.estimated_cost_usd for cr in result.call_records), 8),
        })

        # Annotate captured calls with query context
        new_calls = _captured_calls[call_count_before:]
        cr_list = result.call_records
        for i, (raw, cr) in enumerate(zip(new_calls, cr_list)):
            all_call_records.append({
                "query_id":       result.query_id,
                "query_label":    label,
                "call_index":     i,
                "call_purpose":   cr.call_purpose,
                "timestamp":      raw["timestamp"],
                "input_tokens":   raw["input_tokens"],
                "output_tokens":  raw["output_tokens"],
                "stop_reason":    raw["stop_reason"],
                "raw_response_text": raw["text"],
                "estimated_cost_usd": cr.estimated_cost_usd,
            })

        print(f"Route:   {result.classification}")
        print(f"Refused: {result.refused}")
        if result.refused:
            print(f"Reason:  {result.refusal_reason}")
        else:
            answer_preview = (result.final_answer or "")[:400]
            print(f"Answer:  {answer_preview}")
        print(f"Valid:   {result.validation.valid}  errors={result.validation.errors[:2]}  warnings={result.validation.warnings[:2]}")
        print(f"Cost:    ${record['total_cost_usd']:.6f}  ({record['total_input_tokens']} in / {record['total_output_tokens']} out tokens)")
        print(f"QueryID: {result.query_id}")

    except Exception as exc:
        record["error"] = str(exc)
        print(f"EXCEPTION: {exc}")
        import traceback
        traceback.print_exc()

    all_query_records.append(record)

# ── Write evidence ─────────────────────────────────────────────────────────────

with open(CALLS_PATH, "w") as f:
    for rec in all_call_records:
        f.write(json.dumps(rec) + "\n")

with open(RESULTS_PATH, "w") as f:
    for rec in all_query_records:
        f.write(json.dumps(rec) + "\n")

print(f"\n{'='*70}")
print(f"Queries run: {len(all_query_records)}")
print(f"Calls logged: {len(all_call_records)}")
print(f"Results saved: {RESULTS_PATH}")
print(f"Calls saved:   {CALLS_PATH}")
