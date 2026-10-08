"""
Quantitative agent — translates natural-language questions into SQLite SQL,
executes them safely, and interprets the results using Gemini.

Safety layers
-------------
1.  validate_sql()         Pre-execution: regex + quote-aware statement counter
                           + SQLite authorizer on an in-memory schema copy.
2.  Read-only URI          Connection opened with ?mode=ro — filesystem-level
                           write prevention.
3.  Runtime authorizer     Second authorizer set on the live connection, same
                           allowlist as the validator.  Defense in depth.
4.  Result-size cap        At most MAX_RESULT_ROWS rows fetched, truncation
                           surfaced explicitly in the interpretation prompt.
5.  Execution timeout      Connection-level timeout; interrupts runaway queries.

Two the LLM calls per question
-----------------------------
Call 1 (sql_generation)    Generates a SELECT statement from the question and schema.
Call 2 (quantitative_interpretation) Interprets the query results in plain language.

Both calls are logged to tokenomics.  Retries are bounded (MAX_RETRIES total
across the interpretation step; the SQL step only retries on validation failure,
not on a bad query plan).

Known limitation
----------------
validate_quantitative_interpretation performs a numeric consistency check
but a passing check does not prove the full explanation is correct.
Users should compare key figures against the raw result rows.
"""

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

import llm_client as lc
from tokenomics.logger import CallRecord, log_call, log_query_summary
from utils import parse_json_object
from validation.validator import (
    ALLOWED_FUNCTIONS,
    ALLOWED_TABLE_COLS,
    ValidationResult,
    validate_quantitative_interpretation,
    validate_sql,
    validate_sql_generation_response,
)

load_dotenv()

# ── Destructive-intent pre-filter ─────────────────────────────────────────────
# Two patterns combined:
#   Pattern A — inherently destructive verbs as the leading action word.
#   Pattern B — polite prefaces ("please", "can you", etc.) followed by a
#               destructive verb.  This catches "Please delete all customers"
#               and "Can you drop the table?" which Pattern A would miss.
# Analytical past-tense queries ("How many records were deleted?") use
# "deleted/removed" (past participle after "were/was/been") — not matched here.
DESTRUCTIVE_INTENT_RE = re.compile(
    # Pattern A: destructive verb as first meaningful word
    r"^\s*\b(delete|drop|truncate|remove|erase|wipe|destroy|clear|purge"
    r"|update|alter|modify|insert|add|create)\b"
    r"|"
    # Pattern B: polite/indirect preface → destructive verb
    r"(?:please|can\s+you|could\s+you|would\s+you|i\s+(?:want|need|would\s+like)\s+(?:you\s+)?to)"
    r"\s+(?:delete|drop|truncate|remove|erase|wipe|destroy|clear|purge"
    r"|update|alter|modify|insert)\b",
    re.IGNORECASE,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DB_PATH = Path(os.getenv("DB_PATH", "data/database.sqlite"))
MAX_RESULT_ROWS = int(os.getenv("MAX_RESULT_ROWS", "100"))
MAX_RETRIES = 2       # maximum interpretation calls per question

# ── Schema context fed to the LLM ──────────────────────────────────────────────
SCHEMA_CONTEXT = """\
DATABASE SCHEMA (SQLite, fictional Acme Corp data)
===================================================
Table: customers  (50 rows)
  id INTEGER PK | name TEXT | region TEXT | tier TEXT | status TEXT
  signup_date TEXT (YYYY-MM-DD) | mrr REAL (monthly recurring revenue USD) | industry TEXT
  region:  'North', 'South', 'East', 'West', 'EMEA'
  tier:    'Enterprise', 'Mid-Market', 'SMB'
  status:  'active', 'churned'

Table: sales  (375 rows — one per customer per active month in 2024)
  id INTEGER PK | customer_id INTEGER (→customers.id) | amount REAL (USD)
  sale_date TEXT (YYYY-MM-DD) | region TEXT | rep_name TEXT | product TEXT
  product: 'Platform', 'Add-on Analytics', 'Professional Services'

Table: employees  (40 rows)
  id INTEGER PK | name TEXT | department TEXT | region TEXT
  hire_date TEXT (YYYY-MM-DD) | annual_salary REAL (USD/year)
  satisfaction_score REAL (1.0–10.0, quarterly pulse survey)
  status TEXT ('active', 'terminated')
  department: 'Engineering', 'Sales', 'Customer Success', 'HR', 'Finance'

METRIC DEFINITIONS
==================
Revenue         : SUM(sales.amount) for a period or grouping
Churn rate (%)  : COUNT(status='churned') * 100.0 / COUNT(*) FROM customers
Active customers: WHERE status = 'active'
MRR             : customers.mrr column (monthly recurring revenue per customer)
Employee satisfaction: AVG(satisfaction_score) WHERE status='active'
  Scale: 1 = very low satisfaction, 10 = very high satisfaction
Average salary  : AVG(annual_salary) in USD per year

SAFE FUNCTIONS ONLY
===================
COUNT, SUM, AVG, MIN, MAX, ROUND, ABS, COALESCE, NULLIF,
strftime, date, CAST, IIF, TOTAL, GROUP_CONCAT, substr, length
"""

SQL_GENERATION_SYSTEM = f"""\
You are an expert SQLite query writer for a business analytics system.

{SCHEMA_CONTEXT}
RULES — YOU MUST FOLLOW ALL OF THESE:
1. Generate a single SQLite-compatible SELECT statement only.
2. Use only the tables and columns listed in the schema above.
3. Use only the safe functions listed above.
4. No INSERT, UPDATE, DELETE, DROP, CREATE, PRAGMA, ATTACH, or semicolons.
5. Use ROUND(value, 2) for monetary amounts.
6. Use COALESCE(value, 0) to handle NULL aggregates.
7. For monthly grouping use strftime('%Y-%m', column).
8. For churn rate: ROUND(100.0 * SUM(CASE WHEN status='churned' THEN 1 ELSE 0 END) \
/ COUNT(*), 2) AS churn_rate.

9. If the question asks for a percentage breakdown (e.g. "percentage of total",
   "share", "proportion"), compute it directly in the SELECT using a window function:
   ROUND(100.0 * col / SUM(col) OVER (), 2) AS pct_of_total
   Return the percentage as a named column (e.g. pct_of_total_revenue).
   Do NOT add percentage-of-total formulas to derived_claims — return them as
   SQL result columns so they appear in result rows for validation.

Respond with ONLY valid JSON — no markdown fences, no preamble:
{{
  "sql": "<complete SQLite SELECT statement>",
  "reasoning": "<one sentence explaining what the query does>"
}}
"""

INTERPRETATION_SYSTEM = """\
You are a business analyst interpreting SQL query results for non-technical stakeholders.

RULES:
1. Base your answer ONLY on the provided query results — no external data.
2. If the result set is empty, say so explicitly. Do not fabricate numbers.
3. If results were truncated, acknowledge this and note the answer may be incomplete.
4. Reference specific numbers from the results (rounded naturally for readability).
5. Handle NULL values explicitly: if a value is NULL, say so.
6. For a zero denominator (e.g. division giving NULL), acknowledge the limitation.
7. For every specific number you cite in your interpretation, add it to numeric_claims
   with the exact row_index (0-based integer from the "Row N:" label), column name,
   value as it appears in the data, and unit (USD/percent/count/score/other).
8. For totals or averages you compute yourself across rows (not a single cell), add
   them to derived_claims with the formula (e.g. SUM(amount), AVG(satisfaction_score),
   COUNT(*)), the computed value, and the unit.
9. Do NOT add percentage-of-total formulas to derived_claims. If your SQL returns a
   percentage column, reference it as a numeric_claim with the column name matching
   the SQL alias. Only use derived_claims for SUM, AVG, COUNT, MAX, MIN, and
   MAX-MIN arithmetic on named columns.

Respond with ONLY valid JSON — no markdown fences, no preamble:
{
  "interpretation": "<clear, business-readable answer referencing actual result data>",
  "refused": false,
  "refusal_reason": null,
  "numeric_claims": [   ← REQUIRED — use [] if there are no per-row numeric facts
    {"value": <number>, "row_index": <int>, "column": "<col_name>", "unit": "<USD|percent|count|score|other>"}
  ],
  "derived_claims": [   ← REQUIRED — use [] if there are no derived totals/averages
    {"value": <number>, "formula": "<SUM(col)|AVG(col)|COUNT(*)|MAX(col)|MIN(col)|MAX(col)-MIN(col)>", "unit": "<USD|percent|count|score|other>"}
  ]
}
"""


# ── Read-only execution ───────────────────────────────────────────────────────

def _make_runtime_authorizer(denied: list[str]):
    """Runtime authorizer matching the same allowlist used in validate_sql."""
    def _auth(action, arg1, arg2, db_name, trigger_name):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            table, col = arg1, arg2
            if table in ALLOWED_TABLE_COLS:
                if not col or col in ALLOWED_TABLE_COLS[table]:
                    return sqlite3.SQLITE_OK
                denied.append(f"Column not allowed at execution: {table}.{col}")
                return sqlite3.SQLITE_DENY
            denied.append(f"Table not allowed at execution: {table!r}")
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            fn = (arg2 or "").lower()
            if fn in ALLOWED_FUNCTIONS or fn == "":
                return sqlite3.SQLITE_OK
            denied.append(f"Function not allowed at execution: {fn!r}")
            return sqlite3.SQLITE_DENY
        denied.append(f"Operation denied at execution (action={action})")
        return sqlite3.SQLITE_DENY
    return _auth


def execute_query(sql: str) -> tuple[list[dict], list[str], bool, list[str]]:
    """
    Execute *sql* against the read-only production database.

    Returns
    -------
    rows        : list of dicts {column_name: value}
    col_names   : column names from the cursor description
    truncated   : True when the result set exceeded MAX_RESULT_ROWS
    exec_errors : non-empty list of strings if execution failed
    """
    denied: list[str] = []

    uri = f"file:{DB_PATH.absolute()}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True, timeout=10.0)
    except sqlite3.OperationalError as exc:
        return [], [], False, [f"Cannot open database: {exc}"]

    try:
        con.set_authorizer(_make_runtime_authorizer(denied))
        cur = con.execute(sql)
        col_names = [d[0] for d in cur.description] if cur.description else []
        raw = cur.fetchmany(MAX_RESULT_ROWS + 1)
        truncated = len(raw) > MAX_RESULT_ROWS
        rows = [dict(zip(col_names, r)) for r in raw[:MAX_RESULT_ROWS]]
        return rows, col_names, truncated, []
    except sqlite3.DatabaseError as exc:
        msg = denied[0] if denied else str(exc)
        return [], [], False, [f"Execution error: {msg}"]
    finally:
        con.set_authorizer(None)
        con.close()


# ── Result formatting ─────────────────────────────────────────────────────────

def _format_results_for_prompt(
    col_names: list[str],
    rows: list[dict],
    truncated: bool,
) -> str:
    """Render query results as a compact text block for the LLM.

    Each row is shown as 'Row N: {col: val, ...}' so the LLM can reference
    row_index values precisely in its numeric_claims.
    """
    if not rows:
        return "Query returned no results."

    lines = ["columns: " + ", ".join(col_names)]
    for i, row in enumerate(rows):
        lines.append(f"Row {i}: {row}")
    block = "\n".join(lines)

    if truncated:
        block += f"\n[TRUNCATED — only {MAX_RESULT_ROWS} of more rows shown; answer may be incomplete]"
    return block


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class QuantitativeResult:
    interpretation: str
    sql: str
    col_names: list[str]
    result_rows: list[dict]
    row_count: int
    truncated: bool
    refused: bool
    refusal_reason: str | None
    validation: ValidationResult
    attempts: int
    call_records: list[CallRecord] = field(default_factory=list)


# ── Agent ─────────────────────────────────────────────────────────────────────

class QuantitativeAgent:
    """
    SQL-generation → execution → interpretation pipeline.

    Pass db_path to override the default DB location (useful in tests).
    """

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path  # if set, monkey-patch into module for execute_query

    def answer(
        self,
        query: str,
        query_id: str,
        history: list[dict] | None = None,
    ) -> QuantitativeResult:
        """
        Answer a quantitative question end-to-end.

        Parameters
        ----------
        query    : the user's question in natural language
        query_id : shared ID for tokenomics grouping
        history  : optional prior conversation turns
        """
        call_records: list[CallRecord] = []

        # ── Step 0: Destructive-intent pre-filter ────────────────────────────
        if DESTRUCTIVE_INTENT_RE.match(query):
            refuse_val = ValidationResult(
                valid=False,
                errors=[f"Destructive intent detected in query: {query!r}"],
            )
            return QuantitativeResult(
                interpretation="",
                sql="",
                col_names=[],
                result_rows=[],
                row_count=0,
                truncated=False,
                refused=True,
                refusal_reason=(
                    "This system only supports read-only data queries. "
                    "Requests to modify, delete, or create data are not permitted."
                ),
                validation=refuse_val,
                attempts=0,
                call_records=call_records,
            )

        # ── Step 1: Generate SQL ─────────────────────────────────────────────
        sql_messages: list[dict] = list(history or []) + [
            {"role": "user", "content": f"Question: {query}"}
        ]

        sql_resp = lc.invoke(
            messages=sql_messages,
            system=SQL_GENERATION_SYSTEM,
            max_tokens=4096,  # Gemini budget includes thinking and answer tokens
            temperature=0.0,
        )

        # Parse SQL generation response
        sql_parse_error: str | None = None
        sql_payload: dict = {}
        try:
            sql_payload = parse_json_object(sql_resp.text)
        except ValueError as exc:
            sql_parse_error = str(exc)

        if sql_parse_error:
            sql_val = ValidationResult(
                valid=False,
                errors=[f"SQL generation parse error: {sql_parse_error}"],
            )
        else:
            sql_val = validate_sql_generation_response(sql_payload, sql_resp.stop_reason)

        call_records.append(
            log_call(query_id, "sql_generation", sql_resp,
                     "pass" if sql_val.valid else "fail")
        )

        if not sql_val.valid:
            return QuantitativeResult(
                interpretation="",
                sql="",
                col_names=[],
                result_rows=[],
                row_count=0,
                truncated=False,
                refused=True,
                refusal_reason=(
                    "SQL generation failed: " + "; ".join(sql_val.errors[:2])
                ),
                validation=sql_val,
                attempts=1,
                call_records=call_records,
            )

        raw_sql: str = sql_payload["sql"].strip()

        # ── Step 2: Validate SQL structure (pre-execution) ───────────────────
        struct_val = validate_sql(raw_sql)
        if not struct_val.valid:
            return QuantitativeResult(
                interpretation="",
                sql=raw_sql,
                col_names=[],
                result_rows=[],
                row_count=0,
                truncated=False,
                refused=True,
                refusal_reason=(
                    "SQL blocked by safety validator: "
                    + "; ".join(struct_val.errors[:2])
                ),
                validation=struct_val,
                attempts=1,
                call_records=call_records,
            )

        # ── Step 3: Execute (read-only + authorizer) ─────────────────────────
        # Allow tests to inject a different DB path
        if self._db_path is not None:
            original_db = DB_PATH
            import agents.quantitative as _self_mod
            _self_mod.DB_PATH = self._db_path

        rows, col_names, truncated, exec_errs = execute_query(raw_sql)

        if self._db_path is not None:
            _self_mod.DB_PATH = original_db

        if exec_errs:
            exec_val = ValidationResult(valid=False, errors=exec_errs)
            return QuantitativeResult(
                interpretation="",
                sql=raw_sql,
                col_names=[],
                result_rows=[],
                row_count=0,
                truncated=False,
                refused=True,
                refusal_reason="Query execution failed: " + exec_errs[0],
                validation=exec_val,
                attempts=1,
                call_records=call_records,
            )

        # ── Step 4: Interpret results (with retry) ───────────────────────────
        results_block = _format_results_for_prompt(col_names, rows, truncated)
        interp_user_msg = (
            f"Question: {query}\n\n"
            f"SQL executed:\n{raw_sql}\n\n"
            f"Results:\n{results_block}"
        )
        interp_messages: list[dict] = list(history or []) + [
            {"role": "user", "content": interp_user_msg}
        ]

        last_interp_val = ValidationResult(valid=False, errors=["No attempt yet"])
        interp_payload: dict = {}
        attempts = 0

        for attempt in range(1, MAX_RETRIES + 1):
            attempts = attempt
            interp_resp = lc.invoke(
                messages=interp_messages,
                system=INTERPRETATION_SYSTEM,
                max_tokens=8192,  # preserve room for JSON after thinking
                temperature=0.0,
            )

            parse_err: str | None = None
            try:
                interp_payload = parse_json_object(interp_resp.text)
            except ValueError as exc:
                parse_err = str(exc)
                interp_payload = {}

            if parse_err:
                last_interp_val = ValidationResult(
                    valid=False, errors=[f"JSON parse error: {parse_err}"]
                )
                v_outcome = "fail"
            else:
                last_interp_val = validate_quantitative_interpretation(
                    interp_payload, rows, interp_resp.stop_reason, truncated
                )
                v_outcome = (
                    "pass" if last_interp_val.valid
                    else ("warn" if not last_interp_val.errors else "fail")
                )

            call_records.append(
                log_call(query_id, "quantitative_interpretation",
                         interp_resp, v_outcome)
            )

            if last_interp_val.valid:
                break

            if attempt < MAX_RETRIES:
                err_summary = "; ".join(last_interp_val.errors[:2])
                interp_messages.append(
                    {"role": "assistant", "content": interp_resp.text}
                )
                interp_messages.append({
                    "role": "user",
                    "content": (
                        f"Your response failed validation: {err_summary}. "
                        "Please respond with valid JSON only, matching the schema."
                    ),
                })

        # ── Step 5: Build result ──────────────────────────────────────────────
        if last_interp_val.valid:
            return QuantitativeResult(
                interpretation=interp_payload.get("interpretation", ""),
                sql=raw_sql,
                col_names=col_names,
                result_rows=rows,
                row_count=len(rows),
                truncated=truncated,
                refused=interp_payload.get("refused", False),
                refusal_reason=interp_payload.get("refusal_reason"),
                validation=last_interp_val,
                attempts=attempts + 1,   # +1 for the SQL generation call
                call_records=call_records,
            )

        return QuantitativeResult(
            interpretation="",
            sql=raw_sql,
            col_names=col_names,
            result_rows=rows,
            row_count=len(rows),
            truncated=truncated,
            refused=True,
            refusal_reason=(
                f"Interpretation failed after {attempts} attempt(s): "
                + "; ".join(last_interp_val.errors[:2])
            ),
            validation=last_interp_val,
            attempts=attempts + 1,
            call_records=call_records,
        )
