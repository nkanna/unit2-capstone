"""
Tokenomics logger — records every LLM call and per-query totals.

Two record types are written to tokenomics_log.jsonl:
  "call"          one entry per actual Bedrock invoke (including retries)
  "query_summary" one entry per user query; aggregates all its call records

Keeping them separate avoids double-counting: the summary is derived from
calls but is never used to re-derive calls.

Pricing assumption (configurable via .env):
    us-east-1 us.anthropic.claude-sonnet-4-6
    Input:  $3.30 per million tokens
    Output: $16.50 per million tokens
    Source: AWS Bedrock pricing page, captured 2024-10 — update if pricing changes.
"""

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Pricing ───────────────────────────────────────────────────────────────────
_INPUT_COST_PER_M = float(os.getenv("COST_PER_M_INPUT", "3.30"))
_OUTPUT_COST_PER_M = float(os.getenv("COST_PER_M_OUTPUT", "16.50"))
PRICING_ASSUMPTION = (
    f"us-east-1 us.anthropic.claude-sonnet-4-6: "
    f"${_INPUT_COST_PER_M}/M input, ${_OUTPUT_COST_PER_M}/M output (estimate)"
)

# ── Log path ──────────────────────────────────────────────────────────────────
LOG_PATH = Path(os.getenv("TOKENOMICS_LOG", "tokenomics_log.jsonl"))


@dataclass
class CallRecord:
    record_type: str          # always "call"
    query_id: str
    call_purpose: str         # e.g. "qualitative_generation", "routing", "sql_generation"
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    validation_outcome: str   # "pass", "fail", "warn", or "skipped"
    estimated_cost_usd: float
    pricing_assumption: str
    timestamp: str            # ISO-8601 UTC


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        (input_tokens / 1_000_000) * _INPUT_COST_PER_M
        + (output_tokens / 1_000_000) * _OUTPUT_COST_PER_M,
        8,
    )


def _append(record: dict) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_call(
    query_id: str,
    call_purpose: str,
    bedrock_response,           # BedrockResponse from bedrock_client
    validation_outcome: str,    # "pass" | "fail" | "warn" | "skipped"
    model: str | None = None,
) -> CallRecord:
    """
    Record a single Bedrock invoke.  Called once per actual API request,
    including retries — each retry is a separate call record.
    """
    from bedrock_client import _MODEL_ID  # avoids circular import at module level

    record = CallRecord(
        record_type="call",
        query_id=query_id,
        call_purpose=call_purpose,
        model=model or _MODEL_ID,
        input_tokens=bedrock_response.input_tokens,
        output_tokens=bedrock_response.output_tokens,
        stop_reason=bedrock_response.stop_reason,
        validation_outcome=validation_outcome,
        estimated_cost_usd=_estimate_cost(
            bedrock_response.input_tokens, bedrock_response.output_tokens
        ),
        pricing_assumption=PRICING_ASSUMPTION,
        timestamp=_now_iso(),
    )
    _append(asdict(record))
    return record


def log_query_summary(
    query_id: str,
    user_query: str,
    call_records: list[CallRecord],
) -> dict:
    """
    Write one summary record for a completed user query.
    Totals are computed from the supplied call_records list — never
    read back from the log file — so there is no double-counting risk.
    """
    total_input = sum(r.input_tokens for r in call_records)
    total_output = sum(r.output_tokens for r in call_records)
    total_cost = sum(r.estimated_cost_usd for r in call_records)

    summary = {
        "record_type": "query_summary",
        "query_id": query_id,
        "user_query": user_query,
        "call_count": len(call_records),
        "purposes": [r.call_purpose for r in call_records],
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_estimated_cost_usd": round(total_cost, 8),
        "pricing_assumption": PRICING_ASSUMPTION,
        "timestamp": _now_iso(),
    }
    _append(summary)
    return summary


def new_query_id() -> str:
    """Generate a short unique ID for one user query."""
    return uuid.uuid4().hex[:12]
