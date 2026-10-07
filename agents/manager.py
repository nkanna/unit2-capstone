"""
Manager agent — classifies user queries and routes them to the qualitative
and/or quantitative agents, optionally synthesizing a combined answer.

Routes
------
qualitative  : Documentation / policy questions → QualitativeAgent
quantitative : Data / metrics questions         → QuantitativeAgent
both         : Mixed questions                  → both agents + synthesis
unsupported  : Questions outside scope          → polite refusal

Silver: conversation history is maintained across turns (MAX_HISTORY_TURNS=10).
Gold:   see tests/test_manager.py for offline unit tests.
"""

import json
from collections import deque

from utils import parse_json_object
from dataclasses import dataclass, field
from pathlib import Path

import bedrock_client as bc
from agents.qualitative import QualitativeAgent, QualitativeResult
from agents.quantitative import QuantitativeAgent, QuantitativeResult
from tokenomics.logger import CallRecord, log_call, log_query_summary, new_query_id
from validation.validator import (
    ValidationResult,
    validate_classification_response,
    validate_synthesis_response,
)

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 10   # each turn = 1 user + 1 assistant message pair


# ── Session history ───────────────────────────────────────────────────────────

@dataclass
class Turn:
    user_query: str
    final_answer: str
    query_id: str
    classification: str   # route value


class SessionHistory:
    """
    Bounded ring-buffer of conversation turns.

    to_messages() returns a flat list of {role, content} dicts suitable for
    the Bedrock Messages API.  History is passed as context to agents but
    prior assistant answers are NOT treated as ground truth — each turn
    re-retrieves fresh evidence.
    """

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self._max = max_turns
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add(self, turn: Turn) -> None:
        self._turns.append(turn)

    def reset(self) -> None:
        self._turns.clear()

    def to_messages(self) -> list[dict]:
        msgs: list[dict] = []
        for t in self._turns:
            msgs.append({"role": "user", "content": t.user_query})
            msgs.append({"role": "assistant", "content": t.final_answer})
        return msgs

    def __len__(self) -> int:
        return len(self._turns)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ManagerResult:
    final_answer: str
    classification: str
    refused: bool
    refusal_reason: str | None
    qualitative_result: QualitativeResult | None
    quantitative_result: QuantitativeResult | None
    validation: ValidationResult
    query_id: str
    call_records: list[CallRecord] = field(default_factory=list)


# ── Prompts ───────────────────────────────────────────────────────────────────

CLASSIFICATION_SYSTEM = """\
You are a query router for a business intelligence system that has two knowledge sources:

1. DOCUMENTATION (qualitative): Company policies, procedures, employee handbook,
   onboarding guides, IT/security policies, HR policies.
2. DATABASE (quantitative): SQLite database with customers, sales, and employees tables.
   Handles metrics: revenue, churn rate, MRR, satisfaction scores, salaries.

Classify the user query into one of four routes:
- "qualitative"  : Policy/documentation question only
- "quantitative" : Data/metrics question only
- "both"         : Requires both documentation context AND data
- "unsupported"  : Completely outside scope (e.g. general world knowledge, personal advice)

For "both" route, split the query into focused subqueries.

Respond with ONLY valid JSON — no markdown fences, no preamble:
{
  "route": "<qualitative|quantitative|both|unsupported>",
  "qualitative_subquery": "<focused documentation question or null>",
  "quantitative_subquery": "<focused data question or null>",
  "reasoning": "<one sentence explaining the routing decision>"
}
"""

SYNTHESIS_SYSTEM = """\
You are a business intelligence synthesizer. You have received answers from two sources:
- POLICY: Answer from company documentation/policies
- DATA: Answer from the company database

Combine these into a single coherent, accurate answer for the user.

Rules:
1. Prefix policy-based claims with "According to company policy..."
2. Prefix data-based claims with "The data shows..."
3. Do not add information beyond what the two sources provide.
4. If either source refused or has no answer, acknowledge that gap.
5. Be concise — one to three paragraphs maximum.

Respond with ONLY valid JSON — no markdown fences, no preamble:
{
  "synthesis": "<combined, coherent answer>",
  "refused": false,
  "refusal_reason": null
}
"""


# ── Manager ───────────────────────────────────────────────────────────────────

class Manager:
    """
    Orchestrates classification, routing, and optional synthesis.

    Parameters
    ----------
    chroma_path      : Override ChromaDB directory (for tests).
    collection_name  : Override collection name (for tests).
    db_path          : Override SQLite DB path (for tests).
    """

    def __init__(
        self,
        chroma_path: Path | None = None,
        collection_name: str | None = None,
        db_path: Path | None = None,
    ):
        self._qual_agent = QualitativeAgent(
            chroma_path=chroma_path,
            collection_name=collection_name,
        )
        self._quant_agent = QuantitativeAgent(db_path=db_path)
        self._history = SessionHistory()

    # ── Public API ────────────────────────────────────────────────────────────

    def answer(self, query: str) -> ManagerResult:
        """
        Answer a user query end-to-end.

        One query_id is allocated per call; one log_query_summary is written
        at the end (not inside agent answer() methods).
        """
        query_id = new_query_id()
        call_records: list[CallRecord] = []
        history_msgs = self._history.to_messages()

        # ── Step 1: Classify ─────────────────────────────────────────────────
        class_messages = history_msgs + [{"role": "user", "content": query}]
        class_resp = bc.invoke(
            messages=class_messages,
            system=CLASSIFICATION_SYSTEM,
            max_tokens=512,
            temperature=0.0,
        )

        class_payload: dict = {}
        class_parse_err: str | None = None
        try:
            class_payload = parse_json_object(class_resp.text)
        except ValueError as exc:
            class_parse_err = str(exc)

        if class_parse_err:
            class_val = ValidationResult(
                valid=False,
                errors=[f"Classification parse error: {class_parse_err}"],
            )
        else:
            class_val = validate_classification_response(
                class_payload, class_resp.stop_reason
            )

        call_records.append(
            log_call(query_id, "classification", class_resp,
                     "pass" if class_val.valid else "fail")
        )

        if not class_val.valid:
            result = ManagerResult(
                final_answer="",
                classification="error",
                refused=True,
                refusal_reason="Classification failed: " + "; ".join(class_val.errors[:2]),
                qualitative_result=None,
                quantitative_result=None,
                validation=class_val,
                query_id=query_id,
                call_records=call_records,
            )
            log_query_summary(query_id, query, call_records)
            return result

        route = class_payload["route"]
        qual_subquery = class_payload.get("qualitative_subquery") or query
        quant_subquery = class_payload.get("quantitative_subquery") or query

        # ── Step 2: Route ─────────────────────────────────────────────────────
        qual_result: QualitativeResult | None = None
        quant_result: QuantitativeResult | None = None

        if route == "unsupported":
            result = ManagerResult(
                final_answer=(
                    "I can only answer questions about company policies/procedures "
                    "or business data (customers, sales, employees). "
                    "Your question appears to be outside that scope."
                ),
                classification=route,
                refused=True,
                refusal_reason="Query is outside supported scope.",
                qualitative_result=None,
                quantitative_result=None,
                validation=class_val,
                query_id=query_id,
                call_records=call_records,
            )
            log_query_summary(query_id, query, call_records)
            return result

        if route in ("qualitative", "both"):
            qual_result = self._qual_agent.answer(
                qual_subquery, query_id, history=history_msgs
            )
            call_records.extend(qual_result.call_records)

        if route in ("quantitative", "both"):
            quant_result = self._quant_agent.answer(
                quant_subquery, query_id, history=history_msgs
            )
            call_records.extend(quant_result.call_records)

        # ── Step 3: Synthesize (both route only) ─────────────────────────────
        synth_used_fallback = False
        if route == "both":
            final_answer, synth_val, synth_used_fallback = self._synthesize(
                query, qual_result, quant_result, query_id, call_records
            )
            validation = synth_val
        elif route == "qualitative":
            final_answer = (
                qual_result.answer if qual_result and not qual_result.refused
                else (qual_result.refusal_reason if qual_result else "No answer.")
            )
            validation = qual_result.validation if qual_result else class_val
        else:  # quantitative
            final_answer = (
                quant_result.interpretation if quant_result and not quant_result.refused
                else (quant_result.refusal_reason if quant_result else "No answer.")
            )
            validation = quant_result.validation if quant_result else class_val

        # Determine refusal: for single-agent routes, propagate the agent's refusal.
        # For "both", refuse only when both agents refused.
        if route == "qualitative":
            refused = bool(qual_result and qual_result.refused)
            refusal_reason = (qual_result.refusal_reason if refused else None)
        elif route == "quantitative":
            refused = bool(quant_result and quant_result.refused)
            refusal_reason = (quant_result.refusal_reason if refused else None)
        else:  # both — already handled above; neither agent alone refuses the whole turn
            refused = (
                (qual_result.refused if qual_result else True)
                and (quant_result.refused if quant_result else True)
            )
            refusal_reason = None

        # ── Step 4: History + tokenomics ─────────────────────────────────────
        # Add to history when valid and non-refused, OR when the synthesis
        # fallback was used (the fallback itself is composed of validated
        # agent answers, so it is safe to store even if synthesis failed).
        if not refused and (validation.valid or synth_used_fallback):
            self._history.add(Turn(
                user_query=query,
                final_answer=final_answer,
                query_id=query_id,
                classification=route,
            ))
        log_query_summary(query_id, query, call_records)

        return ManagerResult(
            final_answer=final_answer,
            classification=route,
            refused=refused,
            refusal_reason=refusal_reason,
            qualitative_result=qual_result,
            quantitative_result=quant_result,
            validation=validation,
            query_id=query_id,
            call_records=call_records,
        )

    def reset(self) -> None:
        """Clear session history."""
        self._history.reset()

    # ── Internal ──────────────────────────────────────────────────────────────

    _MAX_SYNTHESIS_RETRIES = 2

    def _synthesize(
        self,
        original_query: str,
        qual_result: QualitativeResult | None,
        quant_result: QuantitativeResult | None,
        query_id: str,
        call_records: list[CallRecord],
    ) -> tuple[str, ValidationResult, bool]:
        """
        Returns (answer_text, validation, used_fallback).

        used_fallback is True when all synthesis retries failed and the
        concatenated qual+quant answer was used instead.  The caller uses
        this flag to decide whether to store the turn in history even when
        the synthesis validation is invalid.
        """
        qual_text = (
            qual_result.answer if qual_result and not qual_result.refused
            else ("(Policy agent refused: "
                  + (qual_result.refusal_reason or "no answer") + ")"
                  if qual_result else "(No policy answer)")
        )
        quant_text = (
            quant_result.interpretation if quant_result and not quant_result.refused
            else ("(Data agent refused: "
                  + (quant_result.refusal_reason or "no answer") + ")"
                  if quant_result else "(No data answer)")
        )

        quant_rows = (
            quant_result.result_rows
            if quant_result and not quant_result.refused
            else None
        )
        policy_texts = (
            [c["document"] for c in qual_result.retrieved_chunks
             if isinstance(c.get("document"), str)]
            if qual_result and not qual_result.refused and qual_result.retrieved_chunks
            else None
        )

        synth_user = (
            f"Original question: {original_query}\n\n"
            f"POLICY answer:\n{qual_text}\n\n"
            f"DATA answer:\n{quant_text}"
        )

        last_synth_val = ValidationResult(valid=False, errors=["No synthesis attempt"])
        for attempt in range(self._MAX_SYNTHESIS_RETRIES):
            synth_resp = bc.invoke(
                messages=[{"role": "user", "content": synth_user}],
                system=SYNTHESIS_SYSTEM,
                max_tokens=1024,
                temperature=0.0,
            )

            synth_payload: dict = {}
            synth_err: str | None = None
            try:
                synth_payload = parse_json_object(synth_resp.text)
            except ValueError as exc:
                synth_err = str(exc)

            if synth_err:
                last_synth_val = ValidationResult(
                    valid=False,
                    errors=[f"Synthesis parse error: {synth_err}"],
                )
            else:
                last_synth_val = validate_synthesis_response(
                    synth_payload, synth_resp.stop_reason,
                    quant_result_rows=quant_rows,
                    retrieved_policy_texts=policy_texts,
                )

            call_records.append(
                log_call(query_id, "synthesis", synth_resp,
                         "pass" if last_synth_val.valid else "fail")
            )

            if last_synth_val.valid:
                return synth_payload.get("synthesis", ""), last_synth_val, False

            if attempt < self._MAX_SYNTHESIS_RETRIES - 1:
                err_summary = "; ".join(last_synth_val.errors[:2])
                synth_user = (
                    synth_user
                    + f"\n\nYour previous response failed validation: {err_summary}. "
                    "Only include numbers that appear in the POLICY answer or DATA answer above."
                )

        # All retries exhausted — fall back to concatenated presentation.
        fallback = (
            "[Partial synthesis — policy and data results shown separately]\n\n"
            f"Policy answer: {qual_text}\n\n"
            f"Data answer: {quant_text}"
        )
        return fallback, last_synth_val, True
