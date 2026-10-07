"""
Validation layer for all Claude responses.

Validators return ValidationResult — a simple pass/fail with
structured error and warning lists so callers can decide how to respond.

validate_qualitative_response       JSON shape, citation IDs, stop_reason.
validate_sql                        Blocks writes, multi-statement, unsafe
                                    patterns, unknown tables/columns via both
                                    regex pre-filter and SQLite authorizer.
validate_sql_generation_response    Structure of the SQL-generation JSON from Claude.
validate_quantitative_interpretation
                                    Structure of the interpretation JSON, plus
                                    a numeric consistency check against the
                                    actual result rows.

SQL-aware validation strategy
------------------------------
validate_sql uses two complementary layers:

Layer 1  Regex pre-filter (_UNSAFE_SQL_RE) blocks obviously dangerous keywords
         before any SQLite code runs.

Layer 2  Quote-aware statement counter (_split_statements) detects multiple
         statements more accurately than a raw ";" regex, because it ignores
         semicolons inside string literals.

Layer 3  SQLite authorizer run against an in-memory copy of the schema
         (created once in _get_validation_db).  SQLite's own query compiler
         calls the authorizer for every table read and function invocation
         during PREPARE — this is genuine SQL-level parsing, not string
         matching.  Any table, column, or function not on the allowlist causes
         prepare to fail with "not authorised", which we surface as an error.

All three layers must pass.  The read-only URI used at execution time is a
fourth, independent layer in the quantitative agent itself.

Known limitations
-----------------
* Citation presence does not prove factual grounding.
* Numeric matching in validate_quantitative_interpretation does not prove
  the full explanation is correct — it only confirms numbers appeared in
  the result set.  Both limitations are documented per requirement.
* Prose cross-check (when numeric_claims present): identifies unsupported
  significant numbers in interpretation text, but cannot detect semantically
  wrong values that happen to appear in result_rows.
* Derived claims on truncated results: SUM/AVG/COUNT derived_claims are
  rejected when results are truncated. MIN/MAX claims generate warnings only
  because sort order cannot be guaranteed.
* Synthesis prose: numbers in synthesis are checked against quant result_rows
  with warnings only; policy-derived numbers cannot be automatically verified.
"""

import re
import sqlite3
import threading
from dataclasses import dataclass, field

# ── Allowed schema (single source of truth for authorizer + callers) ──────────
ALLOWED_TABLE_COLS: dict[str, frozenset[str]] = {
    "customers": frozenset({
        "id", "name", "region", "tier", "status", "signup_date", "mrr", "industry",
    }),
    "sales": frozenset({
        "id", "customer_id", "amount", "sale_date", "region", "rep_name", "product",
    }),
    "employees": frozenset({
        "id", "name", "department", "region", "hire_date",
        "annual_salary", "satisfaction_score", "status",
    }),
}

ALLOWED_FUNCTIONS: frozenset[str] = frozenset({
    "count", "sum", "avg", "min", "max", "round", "abs", "coalesce", "nullif",
    "strftime", "date", "substr", "length", "upper", "lower", "trim",
    "total", "group_concat", "cast", "iif", "typeof", "like", "instr",
    "printf", "format",
})

# ── Numeric precision by unit (for structured claim validation) ───────────────
# None means relative tolerance (0.5%) rather than absolute
PRECISION_BY_UNIT: dict[str, float | None] = {
    "USD": 0.01,       # absolute: ±$0.01
    "percent": 0.1,    # absolute: ±0.1 percentage points
    "count": 0.0,      # exact integer match
    "score": 0.05,     # absolute: ±0.05 (e.g. satisfaction score on 1–10)
    "other": None,     # relative: 0.5%
}

# ── Unsafe pattern pre-filter (regex) ─────────────────────────────────────────
_UNSAFE_SQL_RE = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|REPLACE"
    r"|EXEC|EXECUTE|PRAGMA|ATTACH|DETACH|VACUUM|REINDEX|LOAD_EXTENSION)\b"
    r"|--"
    r"|/\*",
    re.IGNORECASE,
)

# ── In-memory validation DB ───────────────────────────────────────────────────
# Created once; authorizer is set/cleared per call (CLI is single-threaded).
_VALIDATION_SCHEMA = """\
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER, name TEXT, region TEXT, tier TEXT, status TEXT,
    signup_date TEXT, mrr REAL, industry TEXT
);
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER, customer_id INTEGER, amount REAL,
    sale_date TEXT, region TEXT, rep_name TEXT, product TEXT
);
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER, name TEXT, department TEXT, region TEXT,
    hire_date TEXT, annual_salary REAL, satisfaction_score REAL, status TEXT
);
"""
_val_con: sqlite3.Connection | None = None
_val_lock = threading.Lock()


def _get_validation_db() -> sqlite3.Connection:
    global _val_con
    with _val_lock:
        if _val_con is None:
            _val_con = sqlite3.connect(":memory:", check_same_thread=False)
            _val_con.executescript(_VALIDATION_SCHEMA)
    return _val_con


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


# ── SQL helpers ───────────────────────────────────────────────────────────────

def _split_statements(sql: str) -> list[str]:
    """
    Split SQL on semicolons while correctly ignoring semicolons inside
    single-quoted string literals and double-quoted identifiers.
    Returns a list of non-empty statement strings.
    """
    parts: list[str] = []
    current: list[str] = []
    in_single = in_double = False
    for ch in sql:
        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
        elif ch == ";" and not in_single and not in_double:
            stmt = "".join(current).strip()
            if stmt:
                parts.append(stmt)
            current = []
        else:
            current.append(ch)
    remainder = "".join(current).strip()
    if remainder:
        parts.append(remainder)
    return parts


def _authorizer_for_validation(denied: list[str]):
    """
    Returns an authorizer that allows only SELECT + approved table reads
    + approved functions.  Any other action populates *denied* and returns
    SQLITE_DENY so sqlite3.execute raises DatabaseError.
    """
    def _auth(action, arg1, arg2, db_name, trigger_name):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            table, col = arg1, arg2
            if table in ALLOWED_TABLE_COLS:
                # col can be "" when SQLite checks table existence or rowid
                if not col or col in ALLOWED_TABLE_COLS[table]:
                    return sqlite3.SQLITE_OK
                denied.append(f"Column not allowed: {table}.{col}")
                return sqlite3.SQLITE_DENY
            denied.append(f"Table not allowed: {table!r}")
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            fn = (arg2 or "").lower()
            if fn in ALLOWED_FUNCTIONS or fn == "":
                return sqlite3.SQLITE_OK
            denied.append(f"Function not allowed: {fn!r}")
            return sqlite3.SQLITE_DENY
        # Deny INSERT, UPDATE, DELETE, CREATE, PRAGMA, ATTACH, etc.
        denied.append(f"Operation not allowed (action={action})")
        return sqlite3.SQLITE_DENY
    return _auth


# ── validate_sql ──────────────────────────────────────────────────────────────

def validate_sql(sql: str) -> ValidationResult:
    """
    Three-layer SQL validation — see module docstring for the full strategy.

    All three layers must pass before a query should be executed:
    1. Regex pre-filter for obviously dangerous keywords and comment markers.
    2. Quote-aware statement counter rejects multiple statements.
    3. SQLite authorizer rejects unknown tables, columns, and functions.
    """
    errors: list[str] = []
    warnings: list[str] = []

    stripped = sql.strip()
    if not stripped:
        return ValidationResult(valid=False, errors=["SQL is empty"])

    # Layer 1 — regex pre-filter
    first_word = stripped.split()[0].upper()
    if first_word != "SELECT":
        errors.append(f"Only SELECT statements are allowed; got '{first_word}'")

    match = _UNSAFE_SQL_RE.search(stripped)
    if match:
        errors.append(
            f"Unsafe keyword or comment marker detected: {match.group()!r} "
            f"at position {match.start()}"
        )

    # Layer 2 — statement count (SQL-aware, handles string literals)
    stmts = _split_statements(stripped)
    if len(stmts) > 1:
        errors.append(
            f"Multiple statements detected ({len(stmts)}); only one SELECT is allowed"
        )

    if re.search(r"SELECT\s+\*", stripped, re.IGNORECASE):
        warnings.append("SELECT * may return more columns than needed")

    # Short-circuit: don't run the authorizer check if basic structure already failed
    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # Layer 3 — SQLite authorizer (SQL-level parsing via in-memory DB)
    denied: list[str] = []
    val_con = _get_validation_db()
    try:
        val_con.set_authorizer(_authorizer_for_validation(denied))
        val_con.execute(stripped).fetchone()
    except sqlite3.DatabaseError as exc:
        auth_msg = denied[0] if denied else str(exc)
        errors.append(f"SQL schema check failed: {auth_msg}")
    finally:
        val_con.set_authorizer(None)

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ── validate_qualitative_response ─────────────────────────────────────────────

def _normalize_ws(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip ends."""
    import re as _re
    return _re.sub(r'\s+', ' ', text).strip()


def validate_qualitative_response(
    payload: dict,
    retrieved_ids: set[str],
    stop_reason: str,
    chunk_texts: dict[str, str] | None = None,
) -> ValidationResult:
    """
    Validate the structured JSON produced by the qualitative agent.

    Parameters
    ----------
    payload       : parsed dict from Claude's JSON output
    retrieved_ids : set of chunk IDs that were actually retrieved and sent
                    to Claude; citations referencing anything outside this
                    set are fabricated
    stop_reason   : value from BedrockResponse.stop_reason
    chunk_texts   : optional dict mapping chunk_id → raw chunk text.
                    When provided, each citation's excerpt is checked as a
                    case-insensitive substring of its chunk's text after
                    whitespace normalization.  Mismatch → hard error.

    Known limitation: Citation excerpt substring matching confirms the text
    appeared in the chunk but does not prove every claim in the answer is
    grounded.  Human review of retrieved_chunks is still required.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if stop_reason != "end_turn":
        errors.append(
            f"Response was truncated or incomplete (stop_reason={stop_reason!r})"
        )

    required = {"answer", "citations", "refused"}
    missing = required - set(payload.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    if not isinstance(payload["answer"], str):
        errors.append(f"'answer' must be str, got {type(payload['answer']).__name__}")
    if not isinstance(payload["citations"], list):
        errors.append(f"'citations' must be list, got {type(payload['citations']).__name__}")
    if not isinstance(payload["refused"], bool):
        errors.append(f"'refused' must be bool, got {type(payload['refused']).__name__}")

    if isinstance(payload["citations"], list):
        for i, cit in enumerate(payload["citations"]):
            if not isinstance(cit, dict):
                errors.append(f"Citation[{i}] must be an object")
                continue
            if "chunk_id" not in cit:
                errors.append(f"Citation[{i}] missing 'chunk_id'")
                continue
            if cit["chunk_id"] not in retrieved_ids:
                errors.append(
                    f"Citation[{i}] references {cit['chunk_id']!r} "
                    "which was NOT in the retrieved context (possible hallucination)"
                )
            if "excerpt" not in cit:
                warnings.append(f"Citation[{i}] has no 'excerpt' — harder to verify grounding")

    if isinstance(payload.get("refused"), bool):
        answer_str = payload.get("answer", "") if isinstance(payload.get("answer"), str) else ""
        if payload["refused"] and answer_str.strip():
            warnings.append("refused=true but 'answer' is non-empty")
        if not payload["refused"] and not answer_str.strip():
            warnings.append("refused=false but 'answer' is empty")

    # Non-refused answers must have at least one citation.
    if (
        isinstance(payload.get("refused"), bool)
        and not payload["refused"]
        and isinstance(payload.get("citations"), list)
        and len(payload["citations"]) == 0
    ):
        errors.append(
            "Non-refused answer must include at least one citation. "
            "Add citations referencing chunks from the retrieved context."
        )

    # Excerpt substring check (when chunk_texts provided).
    # Whitespace is normalised before comparison to prevent false mismatches.
    # Mismatch is a hard error: citation grounding cannot be assumed.
    # Known limitation: a match confirms the text appeared in the chunk but
    # does not prove every claim in the answer is grounded.
    if chunk_texts is not None and isinstance(payload.get("citations"), list):
        for i, cit in enumerate(payload["citations"]):
            if not isinstance(cit, dict):
                continue
            chunk_id = cit.get("chunk_id")
            excerpt = cit.get("excerpt")
            if chunk_id and excerpt and isinstance(excerpt, str):
                chunk_body = chunk_texts.get(chunk_id, "")
                if chunk_body:
                    norm_excerpt = _normalize_ws(excerpt).lower()
                    norm_body = _normalize_ws(chunk_body).lower()
                    if norm_excerpt not in norm_body:
                        errors.append(
                            f"Citation[{i}] excerpt not found in chunk {chunk_id!r}. "
                            "Substring match does not prove every claim is grounded — "
                            "human review of retrieved_chunks is required."
                        )

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ── validate_sql_generation_response ─────────────────────────────────────────

def validate_sql_generation_response(
    payload: dict,
    stop_reason: str,
) -> ValidationResult:
    """Validate the JSON structure returned by Claude's SQL-generation step."""
    errors: list[str] = []
    warnings: list[str] = []

    if stop_reason != "end_turn":
        errors.append(f"SQL generation response truncated (stop_reason={stop_reason!r})")

    if "sql" not in payload:
        errors.append("Missing required field 'sql' in SQL generation response")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    if not isinstance(payload["sql"], str) or not payload["sql"].strip():
        errors.append("Field 'sql' must be a non-empty string")

    if "reasoning" not in payload:
        warnings.append("No 'reasoning' field — consider adding it for auditability")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ── validate_quantitative_interpretation ──────────────────────────────────────

def _extract_numbers(text: str) -> list[float]:
    """
    Extract significant numeric values from natural language text.
    Strips currency ($) and commas before matching.
    Excludes calendar years (2000–2099) to avoid false matches.
    """
    cleaned = text.replace("$", "").replace(",", "").replace("%", "")
    raw = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned)
    result = []
    for s in raw:
        v = float(s)
        if 2000.0 <= v <= 2099.0:
            continue  # calendar year, not a result value
        if v >= 1.0:   # skip trivial numbers like 0, 0.5
            result.append(v)
    return result


def _numbers_close(a: float, b: float, tolerance: float = 0.02) -> bool:
    """True when |a - b| / max(|b|, 1) <= tolerance."""
    return abs(a - b) / max(abs(b), 1.0) <= tolerance


def _compute_derived(formula: str, result_rows: list[dict]) -> float | None:
    """
    Evaluate a simple derived formula string against result_rows.

    Supported single-aggregate forms:
      SUM(col)   — sum of that column across all rows
      AVG(col)   — average of that column across all rows
      COUNT(*)   — number of rows
      MAX(col)   — maximum of that column across all rows
      MIN(col)   — minimum of that column across all rows

    Supported binary forms (two aggregates connected by + or -):
      MAX(col1) - MIN(col2)   e.g. "MAX(avg_score) - MIN(avg_score)"

    Returns None if the formula is not recognised or computation fails.
    """
    if not formula or not result_rows:
        return None
    f = formula.strip().upper()

    def _col_vals(col: str) -> list[float] | None:
        c = col.lower()
        vals = [row.get(c) for row in result_rows if row.get(c) is not None]
        return [float(v) for v in vals] if vals else None

    def _single(fn: str, col: str) -> float | None:
        vals = _col_vals(col)
        if vals is None:
            return None
        if fn == "SUM":
            return sum(vals)
        if fn == "AVG":
            return sum(vals) / len(vals)
        if fn == "MAX":
            return max(vals)
        if fn == "MIN":
            return min(vals)
        return None

    # COUNT(*) special case
    if re.match(r"COUNT\(\s*\*\s*\)$", f):
        return float(len(result_rows))

    # Single-aggregate
    single_m = re.match(r"(SUM|AVG|MAX|MIN)\((\w+)\)$", f)
    if single_m:
        return _single(single_m.group(1), single_m.group(2))

    # Binary: AGG1(col1) OP AGG2(col2)
    binary_m = re.match(
        r"(SUM|AVG|MAX|MIN)\((\w+)\)\s*([-+])\s*(SUM|AVG|MAX|MIN)\((\w+)\)$", f
    )
    if binary_m:
        v1 = _single(binary_m.group(1), binary_m.group(2))
        op = binary_m.group(3)
        v2 = _single(binary_m.group(4), binary_m.group(5))
        if v1 is None or v2 is None:
            return None
        return v1 - v2 if op == "-" else v1 + v2

    return None


def _prose_cross_check(
    interp_str: str,
    numeric_claims: list,
    derived_claims: list,
    result_rows: list[dict],
    errors: list[str],
    warnings: list[str],
) -> None:
    """
    When numeric_claims is explicitly present in the payload, cross-check
    significant prose numbers against covered values.

    A number is "covered" if it matches (within 2% relative tolerance):
    - Any value in a validated numeric_claim
    - Any successfully computable derived_claim
    - Any numeric value directly in result_rows

    Numbers are excluded from the check if they:
    - Are ≤ 10 (too small to discriminate: counts, versions, scores)
    - Are calendar years (2000-2099) — already excluded by _extract_numbers
    - Match an integer ID from any result row

    Uncovered significant prose numbers are hard errors (trigger retry).

    Known limitation: a semantically wrong value that happens to appear
    in result_rows will pass this check.
    """
    prose_nums = _extract_numbers(interp_str)
    # Check all numbers ≥ 1.0 (calendar years already excluded by _extract_numbers).
    # Small numbers like "8 churned" are important findings and must be traceable.
    significant = [n for n in prose_nums if n >= 1.0]
    if not significant:
        return

    # Build the covered set from result_rows
    covered: list[float] = []
    row_ids: set[float] = set()
    for row in result_rows:
        for k, v in row.items():
            if isinstance(v, (int, float)) and v is not None:
                fv = float(v)
                covered.append(fv)
                if k == "id" or k.endswith("_id"):
                    row_ids.add(fv)

    # Add validated claim values
    for claim in numeric_claims:
        if isinstance(claim, dict) and claim.get("value") is not None:
            try:
                covered.append(float(claim["value"]))
            except (TypeError, ValueError):
                pass
    for claim in derived_claims:
        if isinstance(claim, dict) and claim.get("value") is not None:
            try:
                covered.append(float(claim["value"]))
            except (TypeError, ValueError):
                pass

    unsupported = []
    for n in significant:
        # Skip ID-like integers
        if n in row_ids:
            continue
        # Skip if within 2% of any covered value
        if any(_numbers_close(n, c) for c in covered):
            continue
        unsupported.append(n)

    if unsupported:
        errors.append(
            f"Prose interpretation contains {len(unsupported)} number(s) not "
            f"traceable to result_rows or numeric_claims: {unsupported[:5]}. "
            "Add these to numeric_claims or derived_claims, or remove the claim. "
            "Note: this check does not prove semantic correctness."
        )


def validate_quantitative_interpretation(
    payload: dict,
    result_rows: list[dict],
    stop_reason: str,
    is_truncated: bool = False,
) -> ValidationResult:
    """
    Validate Claude's interpretation of SQL results.

    Numeric consistency check
    -------------------------
    Numbers mentioned in the interpretation are extracted and compared
    against values in result_rows.  Numbers that do not match any result
    cell (within 2 % tolerance) are reported as warnings, not errors —
    because Claude may legitimately derive or round values.

    IMPORTANT: A clean numeric check does NOT prove the full explanation
    is correct.  It only confirms that the specific numbers mentioned
    appeared in the result set.  This limitation is documented here and
    must be noted when presenting results to users.

    Empty-result check
    ------------------
    If result_rows is empty and the interpretation contains specific
    numeric claims, that is treated as a hard error (possible fabrication).
    """
    errors: list[str] = []
    warnings: list[str] = []

    if stop_reason != "end_turn":
        errors.append(
            f"Interpretation was truncated (stop_reason={stop_reason!r})"
        )

    required = {"interpretation", "refused"}
    missing = required - set(payload.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    if not isinstance(payload["interpretation"], str):
        errors.append(
            f"'interpretation' must be str, got {type(payload['interpretation']).__name__}"
        )
    if not isinstance(payload["refused"], bool):
        errors.append(
            f"'refused' must be bool, got {type(payload['refused']).__name__}"
        )

    interp_text = payload.get("interpretation", "")
    interp_str = interp_text if isinstance(interp_text, str) else ""

    # Refused with non-empty interpretation — likely wrong
    if isinstance(payload.get("refused"), bool) and payload["refused"] and interp_str.strip():
        warnings.append("refused=true but 'interpretation' is non-empty")

    # The structured-claims schema (numeric_claims key) is required.
    # Its absence is a hard error so Claude cannot bypass structured validation.
    if "numeric_claims" not in payload:
        errors.append(
            "Missing required field 'numeric_claims'. "
            "Respond with numeric_claims as a list (use [] if there are no numeric facts)."
        )
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # Key presence (even with empty list) signals the new schema — prose cross-check applies.
    claims_schema_present = True
    numeric_claims = payload.get("numeric_claims") or []
    derived_claims = payload.get("derived_claims") or []
    has_structured_claims = bool(numeric_claims or derived_claims)

    if not result_rows:
        if has_structured_claims:
            errors.append(
                "Query returned 0 rows but structured numeric_claims are present. "
                "This may be fabricated."
            )
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        claimed_nums = _extract_numbers(interp_str)
        if claimed_nums:
            errors.append(
                f"Query returned 0 rows but interpretation contains numeric claims "
                f"{claimed_nums[:5]}. This may be fabricated."
            )
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # Truncation acknowledgement check
    if is_truncated and "truncat" not in interp_str.lower():
        warnings.append(
            "Results were truncated but interpretation does not acknowledge this; "
            "the answer may not reflect all data"
        )

    # ── Structured claim validation ──────────────────────────────────────────
    # Validate individual claim objects only when the lists are non-empty.
    # Prose cross-check runs whenever the numeric_claims key was present (even empty).
    if result_rows and has_structured_claims:
        # Validate numeric_claims — each must match the referenced cell
        for i, claim in enumerate(numeric_claims):
            if not isinstance(claim, dict):
                errors.append(f"numeric_claims[{i}] must be an object")
                continue
            claimed_val = claim.get("value")
            row_idx = claim.get("row_index")
            col = claim.get("column")
            unit = claim.get("unit", "other") or "other"
            if claimed_val is None or row_idx is None or col is None:
                errors.append(
                    f"numeric_claims[{i}] missing required fields "
                    f"(value, row_index, column); got {claim}"
                )
                continue
            if not isinstance(row_idx, int) or row_idx < 0 or row_idx >= len(result_rows):
                errors.append(
                    f"numeric_claims[{i}] row_index={row_idx} out of range "
                    f"(result has {len(result_rows)} rows)"
                )
                continue
            actual = result_rows[row_idx].get(col)
            if actual is None:
                errors.append(
                    f"numeric_claims[{i}] column {col!r} not found or NULL "
                    f"in row {row_idx}"
                )
                continue
            # Choose tolerance from PRECISION_BY_UNIT
            precision = PRECISION_BY_UNIT.get(unit, PRECISION_BY_UNIT["other"])
            try:
                cv = float(claimed_val)
                av = float(actual)
            except (TypeError, ValueError):
                errors.append(
                    f"numeric_claims[{i}] non-numeric value: claimed={claimed_val!r}, "
                    f"actual={actual!r}"
                )
                continue
            if precision == 0.0:
                # Exact integer match
                if int(round(cv)) != int(round(av)):
                    errors.append(
                        f"numeric_claims[{i}] count mismatch: claimed {cv} != actual {av} "
                        f"at row {row_idx}[{col!r}]"
                    )
            elif precision is not None:
                # Absolute tolerance
                if abs(cv - av) > precision + 1e-9:
                    errors.append(
                        f"numeric_claims[{i}] value mismatch (unit={unit!r}, "
                        f"tolerance={precision}): claimed {cv} vs actual {av} "
                        f"at row {row_idx}[{col!r}]"
                    )
            else:
                # Relative 0.5% tolerance
                if not _numbers_close(cv, av, tolerance=0.005):
                    errors.append(
                        f"numeric_claims[{i}] value mismatch (relative 0.5%): "
                        f"claimed {cv} vs actual {av} at row {row_idx}[{col!r}]"
                    )

        # Validate derived_claims — compute in Python and compare
        _TRUNCATION_BLOCKED = re.compile(r"^(SUM|AVG|COUNT)\s*\(", re.IGNORECASE)
        _TRUNCATION_WARNED = re.compile(r"^(MAX|MIN)\s*\(", re.IGNORECASE)
        for i, claim in enumerate(derived_claims):
            if not isinstance(claim, dict):
                errors.append(f"derived_claims[{i}] must be an object")
                continue
            claimed_val = claim.get("value")
            formula = claim.get("formula", "")
            unit = claim.get("unit", "other") or "other"
            if claimed_val is None or not formula:
                errors.append(
                    f"derived_claims[{i}] missing 'value' or 'formula'; got {claim}"
                )
                continue
            # Truncation guards: SUM/AVG/COUNT on partial rows gives wrong totals
            if is_truncated:
                if _TRUNCATION_BLOCKED.match(formula.strip()):
                    errors.append(
                        f"derived_claims[{i}] formula {formula!r} cannot be verified: "
                        f"results were truncated at {len(result_rows)} rows — "
                        "the computed total would be incomplete. Re-run without LIMIT "
                        "or increase MAX_RESULT_ROWS."
                    )
                    continue
                if _TRUNCATION_WARNED.match(formula.strip()):
                    warnings.append(
                        f"derived_claims[{i}] formula {formula!r}: MAX/MIN on "
                        "truncated results may not reflect the true extremes if rows "
                        "are not sorted by this column."
                    )
            computed = _compute_derived(formula, result_rows)
            if computed is None:
                errors.append(
                    f"derived_claims[{i}] formula {formula!r} could not be "
                    "computed in Python — unverifiable derived claims are not permitted. "
                    "Use SUM(col), AVG(col), COUNT(*), MAX(col), MIN(col), or "
                    "binary forms like MAX(col) - MIN(col)."
                )
                continue
            precision = PRECISION_BY_UNIT.get(unit, PRECISION_BY_UNIT["other"])
            try:
                cv = float(claimed_val)
            except (TypeError, ValueError):
                errors.append(
                    f"derived_claims[{i}] non-numeric claimed value: {claimed_val!r}"
                )
                continue
            if precision == 0.0:
                if int(round(cv)) != int(round(computed)):
                    errors.append(
                        f"derived_claims[{i}] count mismatch: claimed {cv} != "
                        f"computed {computed} for formula {formula!r}"
                    )
            elif precision is not None:
                if abs(cv - computed) > precision + 1e-9:
                    errors.append(
                        f"derived_claims[{i}] mismatch (unit={unit!r}, "
                        f"tolerance={precision}): claimed {cv} vs computed "
                        f"{computed:.4f} for formula {formula!r}"
                    )
            else:
                if not _numbers_close(cv, computed, tolerance=0.005):
                    errors.append(
                        f"derived_claims[{i}] mismatch (relative 0.5%): claimed "
                        f"{cv} vs computed {computed:.4f} for formula {formula!r}"
                    )

    # Prose cross-check: fires whenever the numeric_claims key was present in the
    # payload (even when the list is empty).  Any significant prose number must be
    # traceable to result_rows or to a validated claim.
    if result_rows and claims_schema_present:
        _prose_cross_check(
            interp_str, numeric_claims, derived_claims, result_rows, errors, warnings
        )

    elif result_rows and not claims_schema_present:
        # Fall back to soft text-scan for responses without structured claims
        claimed_nums = _extract_numbers(interp_str)
        if claimed_nums:
            result_nums: list[float] = []
            for row in result_rows:
                for val in row.values():
                    if isinstance(val, (int, float)) and val is not None:
                        result_nums.append(float(val))
            unmatched = [
                n for n in claimed_nums
                if n >= 1.0
                and not any(_numbers_close(n, r) for r in result_nums)
            ]
            if unmatched:
                warnings.append(
                    f"Claimed numbers not found in result data: {unmatched[:5]}. "
                    "Verify manually. Note: passing this check does NOT prove the "
                    "full explanation is correct."
                )

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ── validate_classification_response ─────────────────────────────────────────

def validate_classification_response(
    payload: dict,
    stop_reason: str,
) -> ValidationResult:
    """
    Validate the classification JSON returned by the Manager.

    Expected schema:
      {
        "route": "qualitative" | "quantitative" | "both" | "unsupported",
        "qualitative_subquery": <str or null>,
        "quantitative_subquery": <str or null>,
        "reasoning": <str>
      }
    """
    errors: list[str] = []
    warnings: list[str] = []

    if stop_reason != "end_turn":
        errors.append(f"Classification response truncated (stop_reason={stop_reason!r})")

    required = {"route"}
    missing = required - set(payload.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    valid_routes = {"qualitative", "quantitative", "both", "unsupported"}
    route = payload.get("route", "")
    if route not in valid_routes:
        errors.append(
            f"'route' must be one of {sorted(valid_routes)}, got {route!r}"
        )

    if route == "qualitative" and not payload.get("qualitative_subquery"):
        warnings.append("route=qualitative but qualitative_subquery is empty")
    if route == "quantitative" and not payload.get("quantitative_subquery"):
        warnings.append("route=quantitative but quantitative_subquery is empty")
    if route == "both":
        if not payload.get("qualitative_subquery"):
            errors.append("route=both requires qualitative_subquery")
        if not payload.get("quantitative_subquery"):
            errors.append("route=both requires quantitative_subquery")

    if "reasoning" not in payload:
        warnings.append("No 'reasoning' field — classification is not auditable")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ── validate_synthesis_response ───────────────────────────────────────────────

def validate_synthesis_response(
    payload: dict,
    stop_reason: str,
    quant_result_rows: list[dict] | None = None,
    retrieved_policy_texts: list[str] | None = None,
) -> ValidationResult:
    """
    Validate the synthesis JSON returned by the Manager when route=both.

    Expected schema:
      {
        "synthesis": "<combined answer>",
        "refused": false
      }

    When evidence is provided, every significant number (≥ 1.0, excluding
    calendar years) in the synthesis text must be traceable to either
    quant_result_rows or retrieved_policy_texts.  Untraceable numbers are
    hard errors.

    quant_result_rows  : SQL result rows; numbers in synthesis must appear here
                         or in policy texts.
    retrieved_policy_texts : raw chunk texts from the qualitative agent.
                         Numbers that appear in these texts are considered
                         policy-grounded and are allowed.

    When both parameters are None, no numeric grounding check is performed.

    Known limitation: grounding check does not prove semantic correctness;
    it only confirms numbers appeared in the evidence sources.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if stop_reason != "end_turn":
        errors.append(f"Synthesis response truncated (stop_reason={stop_reason!r})")

    required = {"synthesis", "refused"}
    missing = required - set(payload.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    if not isinstance(payload["synthesis"], str):
        errors.append(
            f"'synthesis' must be str, got {type(payload['synthesis']).__name__}"
        )
    if not isinstance(payload["refused"], bool):
        errors.append(
            f"'refused' must be bool, got {type(payload['refused']).__name__}"
        )

    synth_str = payload.get("synthesis", "")
    if isinstance(synth_str, str) and not payload.get("refused") and not synth_str.strip():
        warnings.append("refused=false but 'synthesis' is empty")

    # Synthesis refusal is a hard error — triggers retry/fallback in _synthesize.
    if payload.get("refused"):
        reason = payload.get("refusal_reason") or "no reason given"
        errors.append(f"Synthesis refused: {reason}")

    # Hard grounding check when at least one evidence source is provided.
    evidence_provided = (quant_result_rows is not None or retrieved_policy_texts is not None)
    if evidence_provided and isinstance(synth_str, str):
        prose_nums = _extract_numbers(synth_str)
        significant = [n for n in prose_nums if n >= 1.0]
        if significant:
            # Covered by quant result rows
            covered: list[float] = []
            if quant_result_rows:
                for row in quant_result_rows:
                    for v in row.values():
                        if isinstance(v, (int, float)) and v is not None:
                            covered.append(float(v))

            # Covered by policy texts: extract all numbers from raw chunk strings
            policy_covered: set[float] = set()
            if retrieved_policy_texts:
                for text in retrieved_policy_texts:
                    clean = text.replace(",", "").replace("$", "").replace("%", "")
                    for m in re.finditer(r'\b(\d+(?:\.\d+)?)\b', clean):
                        try:
                            v = float(m.group(1))
                            if not (2000.0 <= v <= 2099.0):
                                policy_covered.add(v)
                        except ValueError:
                            pass

            unsupported = []
            for n in significant:
                if any(_numbers_close(n, c) for c in covered):
                    continue
                if any(_numbers_close(n, pc) for pc in policy_covered):
                    continue
                unsupported.append(n)

            if unsupported:
                errors.append(
                    f"Synthesis contains {len(unsupported)} number(s) not traceable to "
                    f"quantitative result rows or retrieved policy text: {unsupported[:5]}. "
                    "These may be fabricated. Add them to the evidence or remove the claim. "
                    "Note: this check does not prove semantic correctness."
                )

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
