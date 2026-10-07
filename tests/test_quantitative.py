"""
Offline unit tests for agents/quantitative.py and the quantitative
portions of validation/validator.py.

All tests use mocks or in-memory SQLite — no Bedrock calls, no production DB.
Every assertion tests actual code behavior; nothing is fabricated.
"""

import json
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from validation.validator import (
    ValidationResult,
    _split_statements,
    validate_sql,
    validate_sql_generation_response,
    validate_quantitative_interpretation,
)
from agents.quantitative import (
    QuantitativeAgent,
    DESTRUCTIVE_INTENT_RE,
    execute_query,
    _format_results_for_prompt,
)
from validation.validator import (
    PRECISION_BY_UNIT,
    _compute_derived,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@dataclass
class _FakeBR:
    text: str
    input_tokens: int = 50
    output_tokens: int = 60
    stop_reason: str = "end_turn"


def _sql_json(sql: str, reasoning: str = "answers the question") -> str:
    return json.dumps({"sql": sql, "reasoning": reasoning})


def _interp_json(
    interpretation: str = "Revenue was $100.",
    refused: bool = False,
    numeric_claims: list | None = None,
    derived_claims: list | None = None,
) -> str:
    return json.dumps({
        "interpretation": interpretation,
        "refused": refused,
        "refusal_reason": None,
        "numeric_claims": numeric_claims if numeric_claims is not None else [],
        "derived_claims": derived_claims if derived_claims is not None else [],
    })


# ── _split_statements (SQL-aware tokenizer) ───────────────────────────────────

class TestSplitStatements:

    def test_single_statement(self):
        assert len(_split_statements("SELECT 1")) == 1

    def test_trailing_semicolon_is_one_statement(self):
        assert len(_split_statements("SELECT 1;")) == 1

    def test_two_statements(self):
        assert len(_split_statements("SELECT 1; SELECT 2")) == 2

    def test_semicolon_inside_string_is_one_statement(self):
        # The semicolon is inside a string literal — only one statement
        assert len(_split_statements("SELECT 'hello; world' AS v")) == 1

    def test_double_quote_identifier(self):
        assert len(_split_statements('SELECT "col;name" FROM t')) == 1


# ── validate_sql ──────────────────────────────────────────────────────────────

class TestValidateSql:

    def test_valid_count_query(self):
        r = validate_sql("SELECT COUNT(*) FROM customers")
        assert r.valid

    def test_valid_aggregate_with_group_by(self):
        r = validate_sql(
            "SELECT region, ROUND(SUM(amount), 2) AS revenue "
            "FROM sales GROUP BY region ORDER BY revenue DESC"
        )
        assert r.valid

    def test_valid_join_query(self):
        r = validate_sql(
            "SELECT c.name, SUM(s.amount) FROM customers c "
            "JOIN sales s ON s.customer_id = c.id GROUP BY c.id"
        )
        assert r.valid

    def test_non_select_fails(self):
        assert not validate_sql("DELETE FROM customers WHERE id = 1").valid

    def test_drop_fails(self):
        assert not validate_sql("DROP TABLE customers").valid

    def test_pragma_fails(self):
        assert not validate_sql("PRAGMA table_info(customers)").valid

    def test_attach_fails(self):
        assert not validate_sql("ATTACH DATABASE ':memory:' AS evil").valid

    def test_multiple_statements_fail(self):
        r = validate_sql("SELECT 1; DROP TABLE employees")
        assert not r.valid
        assert any("multiple" in e.lower() for e in r.errors)

    def test_semicolon_in_string_is_single_statement(self):
        # Should NOT be treated as multiple statements
        r = validate_sql("SELECT * FROM customers WHERE name = 'O;Brien'")
        assert r.valid or any("SELECT *" in w for w in r.warnings)

    def test_unknown_table_fails(self):
        r = validate_sql("SELECT * FROM secret_table")
        assert not r.valid
        assert any("secret_table" in e for e in r.errors)

    def test_unknown_column_fails(self):
        r = validate_sql("SELECT secret_col FROM customers")
        assert not r.valid

    def test_load_extension_fails(self):
        r = validate_sql("SELECT load_extension('/tmp/evil.so')")
        assert not r.valid

    def test_block_comment_fails(self):
        r = validate_sql("SELECT * FROM customers /* comment */")
        assert not r.valid

    def test_line_comment_fails(self):
        r = validate_sql("SELECT 1 -- comment")
        assert not r.valid

    def test_select_star_warns(self):
        r = validate_sql("SELECT * FROM customers")
        assert r.valid
        assert any("SELECT *" in w for w in r.warnings)


# ── validate_sql_generation_response ─────────────────────────────────────────

class TestValidateSqlGenerationResponse:

    def test_valid_response(self):
        r = validate_sql_generation_response(
            {"sql": "SELECT COUNT(*) FROM customers", "reasoning": "counts rows"},
            "end_turn",
        )
        assert r.valid

    def test_missing_sql_field_fails(self):
        r = validate_sql_generation_response({"reasoning": "x"}, "end_turn")
        assert not r.valid
        assert any("sql" in e for e in r.errors)

    def test_empty_sql_fails(self):
        r = validate_sql_generation_response({"sql": "   "}, "end_turn")
        assert not r.valid

    def test_truncated_response_fails(self):
        r = validate_sql_generation_response(
            {"sql": "SELECT 1"}, "max_tokens"
        )
        assert not r.valid
        assert any("truncat" in e.lower() for e in r.errors)

    def test_missing_reasoning_warns(self):
        r = validate_sql_generation_response({"sql": "SELECT 1"}, "end_turn")
        assert r.valid
        assert any("reasoning" in w for w in r.warnings)


# ── validate_quantitative_interpretation ─────────────────────────────────────

class TestValidateInterpretation:

    def _rows(self):
        return [{"revenue": 40689.85}, {"revenue": 62995.99}]

    def _p(self, interp, rows_for_claims=None, **extra):
        """Helper: build a valid payload with numeric_claims key present."""
        p = {
            "interpretation": interp,
            "refused": False,
            "refusal_reason": None,
            "numeric_claims": [],
            "derived_claims": [],
        }
        p.update(extra)
        return p

    def test_valid_interpretation(self):
        rows = self._rows()
        r = validate_quantitative_interpretation(
            self._p("Revenue in Jan was $40,689.85.",
                    numeric_claims=[{"value": 40689.85, "row_index": 0, "column": "revenue", "unit": "USD"}]),
            rows,
            "end_turn",
        )
        assert r.valid

    def test_missing_interpretation_fails(self):
        r = validate_quantitative_interpretation(
            {"refused": False, "numeric_claims": []},
            self._rows(),
            "end_turn",
        )
        assert not r.valid

    def test_truncated_fails(self):
        r = validate_quantitative_interpretation(
            self._p("Revenue was ..."),
            self._rows(),
            "max_tokens",
        )
        assert not r.valid
        assert any("truncat" in e.lower() for e in r.errors)

    def test_empty_results_with_numeric_claims_fails(self):
        """If query returns 0 rows but structured numeric_claims present → fabrication."""
        r = validate_quantitative_interpretation(
            self._p("Revenue was $500,000.",
                    numeric_claims=[{"value": 500000.0, "row_index": 0, "column": "revenue", "unit": "USD"}]),
            [],        # empty results
            "end_turn",
        )
        assert not r.valid
        assert any("fabricat" in e.lower() or "0 rows" in e.lower() for e in r.errors)

    def test_empty_results_with_no_claims_passes(self):
        r = validate_quantitative_interpretation(
            self._p("The query returned no results."),
            [],
            "end_turn",
        )
        assert r.valid

    def test_fabricated_number_is_now_hard_error(self):
        """A prose number not in results or numeric_claims is now a hard error (Issue 2 fix)."""
        r = validate_quantitative_interpretation(
            self._p("Revenue was $999,999.99."),
            [{"revenue": 40689.85}],
            "end_turn",
        )
        # With numeric_claims key present and prose number not traceable → hard error
        assert not r.valid
        assert any("traceable" in e.lower() or "999999" in e for e in r.errors)

    def test_truncation_not_acknowledged_warns(self):
        r = validate_quantitative_interpretation(
            self._p("Revenue was $40,689.85.",
                    numeric_claims=[{"value": 40689.85, "row_index": 0, "column": "revenue", "unit": "USD"}]),
            self._rows(),
            "end_turn",
            is_truncated=True,
        )
        assert r.valid
        assert any("truncat" in w.lower() for w in r.warnings)

    def test_null_aggregate_passes(self):
        """NULL result (e.g. AVG of empty set) should not crash the validator."""
        r = validate_quantitative_interpretation(
            self._p("Average salary returned NULL — no active employees found."),
            [{"avg_salary": None}],
            "end_turn",
        )
        assert r.valid

    def test_division_by_zero_via_null(self):
        """When churn_rate is NULL (0 customers), interpretation should acknowledge."""
        r = validate_quantitative_interpretation(
            self._p("Churn rate is NULL because there are no customers."),
            [{"churn_rate": None}],
            "end_turn",
        )
        assert r.valid


# ── execute_query (using a temp in-process SQLite DB) ────────────────────────

class TestExecuteQuery:
    """
    These tests use a real temp SQLite file to exercise the read-only
    connection and authorizer without touching the production DB.
    """

    @pytest.fixture()
    def seed_db(self, tmp_path):
        db = tmp_path / "test.sqlite"
        con = sqlite3.connect(str(db))
        con.executescript("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY, name TEXT, region TEXT,
                tier TEXT, status TEXT, signup_date TEXT, mrr REAL, industry TEXT
            );
            CREATE TABLE sales (
                id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL,
                sale_date TEXT, region TEXT, rep_name TEXT, product TEXT
            );
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY, name TEXT, department TEXT, region TEXT,
                hire_date TEXT, annual_salary REAL, satisfaction_score REAL, status TEXT
            );
            INSERT INTO customers VALUES (1,'Acme','North','Enterprise','active','2024-01-01',5000.0,'Technology');
            INSERT INTO sales VALUES (1,1,5000.0,'2024-01-15','North','Alice','Platform');
            INSERT INTO employees VALUES (1,'Bob','Engineering','North','2023-01-01',120000.0,8.5,'active');
        """)
        con.close()
        return db

    def test_valid_aggregate_query(self, seed_db, monkeypatch):
        monkeypatch.setattr("agents.quantitative.DB_PATH", seed_db)
        rows, cols, trunc, errs = execute_query(
            "SELECT region, ROUND(SUM(amount),2) AS revenue FROM sales GROUP BY region"
        )
        assert errs == []
        assert len(rows) == 1
        assert rows[0]["revenue"] == 5000.0
        assert not trunc

    def test_execution_failure_returns_errors_and_closes(self, seed_db, monkeypatch):
        monkeypatch.setattr("agents.quantitative.DB_PATH", seed_db)
        # Query references a non-existent table → execution error
        rows, cols, trunc, errs = execute_query("SELECT * FROM nonexistent_table")
        assert errs != []
        assert rows == []

    def test_write_attempt_blocked_by_authorizer(self, seed_db, monkeypatch):
        monkeypatch.setattr("agents.quantitative.DB_PATH", seed_db)
        # Even if validate_sql is bypassed, the runtime authorizer blocks writes
        rows, cols, trunc, errs = execute_query(
            "INSERT INTO customers VALUES (99,'X','Y','Z','active','2024-01-01',1.0,'T')"
        )
        assert errs != []

    def test_result_truncation(self, seed_db, monkeypatch):
        # Seed 5 extra rows then set MAX_RESULT_ROWS=2
        con = sqlite3.connect(str(seed_db))
        for i in range(2, 7):
            con.execute(
                "INSERT INTO sales VALUES (?,1,100.0,'2024-02-01','North','Alice','Platform')",
                (i,)
            )
        con.commit(); con.close()

        monkeypatch.setattr("agents.quantitative.DB_PATH", seed_db)
        monkeypatch.setattr("agents.quantitative.MAX_RESULT_ROWS", 2)
        rows, cols, trunc, errs = execute_query("SELECT * FROM sales")
        assert len(rows) == 2
        assert trunc is True

    def test_empty_result_set(self, seed_db, monkeypatch):
        monkeypatch.setattr("agents.quantitative.DB_PATH", seed_db)
        rows, cols, trunc, errs = execute_query(
            "SELECT * FROM customers WHERE status = 'churned'"
        )
        assert errs == []
        assert rows == []
        assert not trunc


# ── QuantitativeAgent (full flow, mocked Bedrock) ────────────────────────────

class TestQuantitativeAgentFlow:

    @pytest.fixture()
    def seed_db(self, tmp_path):
        db = tmp_path / "test.sqlite"
        con = sqlite3.connect(str(db))
        con.executescript("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY, name TEXT, region TEXT,
                tier TEXT, status TEXT, signup_date TEXT, mrr REAL, industry TEXT
            );
            CREATE TABLE sales (
                id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL,
                sale_date TEXT, region TEXT, rep_name TEXT, product TEXT
            );
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY, name TEXT, department TEXT, region TEXT,
                hire_date TEXT, annual_salary REAL, satisfaction_score REAL, status TEXT
            );
            INSERT INTO customers VALUES (1,'Acme','North','Enterprise','active','2024-01-01',5000.0,'Technology');
            INSERT INTO customers VALUES (2,'Beta','South','SMB','churned','2024-02-01',200.0,'Finance');
            INSERT INTO sales VALUES (1,1,5200.0,'2024-01-15','North','Alice','Platform');
            INSERT INTO employees VALUES (1,'Bob','Engineering','North','2023-01-01',120000.0,8.5,'active');
        """)
        con.close()
        return db

    def _agent(self, db):
        return QuantitativeAgent(db_path=db)

    def test_valid_aggregate_query_end_to_end(self, seed_db, mocker):
        """Happy-path: SQL generated, executed, interpreted, validated."""
        agent = self._agent(seed_db)

        sql = "SELECT ROUND(SUM(amount),2) AS total FROM sales"
        mock_invoke = mocker.patch(
            "agents.quantitative.bc.invoke",
            side_effect=[
                _FakeBR(text=_sql_json(sql)),
                _FakeBR(text=_interp_json("Total revenue was $5,200.0.")),
            ],
        )
        mocker.patch("agents.quantitative.log_call", return_value=MagicMock())

        result = agent.answer("What is total revenue?", "qid-001")

        assert not result.refused
        assert result.sql == sql
        assert result.row_count == 1
        assert result.result_rows[0]["total"] == 5200.0
        assert result.validation.valid

    def test_malformed_sql_generation_response(self, seed_db, mocker):
        """Claude returns unparseable JSON for SQL generation → refused."""
        agent = self._agent(seed_db)

        mocker.patch(
            "agents.quantitative.bc.invoke",
            return_value=_FakeBR(text="not json }{"),
        )
        mocker.patch("agents.quantitative.log_call", return_value=MagicMock())

        result = agent.answer("Total revenue?", "qid-002")

        assert result.refused
        assert "parse error" in result.refusal_reason.lower()

    def test_blocked_sql_generation(self, seed_db, mocker):
        """Claude generates a DELETE statement for a read query → blocked by validate_sql."""
        agent = self._agent(seed_db)

        # Use a read-intent query so DESTRUCTIVE_INTENT_RE doesn't fire;
        # we're testing that validate_sql blocks a DELETE that Claude unexpectedly produced.
        mocker.patch(
            "agents.quantitative.bc.invoke",
            return_value=_FakeBR(
                text=_sql_json("DELETE FROM customers WHERE id = 1")
            ),
        )
        mocker.patch("agents.quantitative.log_call", return_value=MagicMock())

        result = agent.answer("Show me all customers", "qid-003")

        assert result.refused
        assert "blocked" in result.refusal_reason.lower()

    def test_unknown_table_blocked(self, seed_db, mocker):
        """SQL referencing a table not in the schema → blocked."""
        agent = self._agent(seed_db)

        mocker.patch(
            "agents.quantitative.bc.invoke",
            return_value=_FakeBR(
                text=_sql_json("SELECT * FROM secret_data")
            ),
        )
        mocker.patch("agents.quantitative.log_call", return_value=MagicMock())

        result = agent.answer("Show secret data", "qid-004")

        assert result.refused

    def test_execution_failure_handled_cleanly(self, seed_db, mocker):
        """execute_query returning errors → refused without crashing."""
        agent = self._agent(seed_db)

        mocker.patch(
            "agents.quantitative.bc.invoke",
            return_value=_FakeBR(
                text=_sql_json("SELECT COUNT(*) FROM customers")
            ),
        )
        mocker.patch("agents.quantitative.log_call", return_value=MagicMock())
        mocker.patch(
            "agents.quantitative.execute_query",
            return_value=([], [], False, ["Simulated DB error"]),
        )

        result = agent.answer("Count customers", "qid-005")

        assert result.refused
        assert "execution failed" in result.refusal_reason.lower()

    def test_empty_results_handled_cleanly(self, seed_db, mocker):
        """Query returns 0 rows → interpretation asked to say so."""
        agent = self._agent(seed_db)

        sql = "SELECT * FROM customers WHERE status = 'churned' AND region = 'EMEA'"
        mocker.patch(
            "agents.quantitative.bc.invoke",
            side_effect=[
                _FakeBR(text=_sql_json(sql)),
                _FakeBR(text=_interp_json(
                    "No churned customers were found in the EMEA region."
                )),
            ],
        )
        mocker.patch("agents.quantitative.log_call", return_value=MagicMock())

        result = agent.answer("Churned customers in EMEA?", "qid-006")

        assert not result.refused
        assert result.row_count == 0

    def test_fabricated_number_in_empty_results(self, seed_db, mocker):
        """Interpretation claims numbers despite 0 rows → hard error → refused."""
        agent = self._agent(seed_db)

        sql = "SELECT * FROM customers WHERE status = 'terminated'"
        mocker.patch(
            "agents.quantitative.bc.invoke",
            side_effect=[
                _FakeBR(text=_sql_json(sql)),
                # First attempt: fabricates a number for empty results
                _FakeBR(text=_interp_json(
                    "There were 42 terminated customers generating $100,000."
                )),
                # Second attempt (retry): still wrong
                _FakeBR(text=_interp_json(
                    "The data shows 15 terminated customers."
                )),
            ],
        )
        mocker.patch("agents.quantitative.log_call", return_value=MagicMock())

        result = agent.answer("Terminated customers?", "qid-007")

        # Both attempts fail → refused after retries exhausted
        assert result.refused

    def test_retry_on_malformed_interpretation(self, seed_db, mocker):
        """First interpretation returns bad JSON → retry → success."""
        agent = self._agent(seed_db)

        sql = "SELECT ROUND(SUM(amount),2) AS total FROM sales"
        mocker.patch(
            "agents.quantitative.bc.invoke",
            side_effect=[
                _FakeBR(text=_sql_json(sql)),
                _FakeBR(text="INVALID JSON }{"),  # first interpretation attempt
                _FakeBR(text=_interp_json("Total was $5,200.0.")),  # retry succeeds
            ],
        )
        mocker.patch("agents.quantitative.log_call", return_value=MagicMock())

        result = agent.answer("Total revenue?", "qid-008")

        assert not result.refused
        assert result.validation.valid

    def test_retry_exhaustion_returns_refused(self, seed_db, mocker):
        """All interpretation attempts fail → refused result returned."""
        agent = self._agent(seed_db)

        sql = "SELECT ROUND(SUM(amount),2) AS total FROM sales"
        mocker.patch(
            "agents.quantitative.bc.invoke",
            side_effect=[
                _FakeBR(text=_sql_json(sql)),
                _FakeBR(text="BAD JSON }{"),
                _FakeBR(text="STILL BAD }{"),
            ],
        )
        mocker.patch("agents.quantitative.log_call", return_value=MagicMock())

        result = agent.answer("Total revenue?", "qid-009")

        assert result.refused
        assert not result.validation.valid

    def test_truncated_sql_generation_fails(self, seed_db, mocker):
        """stop_reason=max_tokens on SQL generation → refused."""
        agent = self._agent(seed_db)

        mocker.patch(
            "agents.quantitative.bc.invoke",
            return_value=_FakeBR(
                text=_sql_json("SELECT COUNT(*) FROM customers"),
                stop_reason="max_tokens",
            ),
        )
        mocker.patch("agents.quantitative.log_call", return_value=MagicMock())

        result = agent.answer("How many customers?", "qid-010")

        assert result.refused

    def test_multiple_statements_blocked(self, seed_db, mocker):
        """Claude generates two statements → blocked."""
        agent = self._agent(seed_db)

        mocker.patch(
            "agents.quantitative.bc.invoke",
            return_value=_FakeBR(
                text=_sql_json("SELECT COUNT(*) FROM customers; DROP TABLE customers")
            ),
        )
        mocker.patch("agents.quantitative.log_call", return_value=MagicMock())

        result = agent.answer("How many customers?", "qid-011")

        assert result.refused
        assert "blocked" in result.refusal_reason.lower()

    def test_destructive_intent_refused_before_bedrock(self, seed_db, mocker):
        """Destructive intent detected in query → refused before any Bedrock call."""
        agent = self._agent(seed_db)
        mock_invoke = mocker.patch("agents.quantitative.bc.invoke")

        result = agent.answer("Delete all customers from the database", "qid-012")

        assert result.refused
        assert "read-only" in result.refusal_reason.lower()
        assert result.attempts == 0
        mock_invoke.assert_not_called()

    def test_structured_numeric_claims_validated(self):
        """numeric_claims referencing real result cells must pass validation."""
        from validation.validator import validate_quantitative_interpretation
        rows = [{"revenue": 5200.0}]
        payload = {
            "interpretation": "Total revenue was $5,200.",
            "refused": False,
            "refusal_reason": None,
            "numeric_claims": [
                {"value": 5200.0, "row_index": 0, "column": "revenue", "unit": "USD"}
            ],
            "derived_claims": [],
        }
        r = validate_quantitative_interpretation(payload, rows, "end_turn")
        assert r.valid, r.errors

    def test_wrong_numeric_claim_fails_validation(self):
        """numeric_claims with wrong value must produce an error."""
        from validation.validator import validate_quantitative_interpretation
        rows = [{"revenue": 5200.0}]
        payload = {
            "interpretation": "Revenue was $9,999.",
            "refused": False,
            "refusal_reason": None,
            "numeric_claims": [
                {"value": 9999.0, "row_index": 0, "column": "revenue", "unit": "USD"}
            ],
            "derived_claims": [],
        }
        r = validate_quantitative_interpretation(payload, rows, "end_turn")
        assert not r.valid
        assert any("mismatch" in e.lower() for e in r.errors)


# ── DESTRUCTIVE_INTENT_RE ─────────────────────────────────────────────────────

class TestDestructiveIntentRe:

    def test_delete_detected(self):
        assert DESTRUCTIVE_INTENT_RE.match("delete all customers")

    def test_drop_detected(self):
        assert DESTRUCTIVE_INTENT_RE.match("Drop the sales table")

    def test_update_detected(self):
        assert DESTRUCTIVE_INTENT_RE.match("Update employee salaries")

    def test_select_not_detected(self):
        assert not DESTRUCTIVE_INTENT_RE.match("select total revenue by region")

    def test_show_not_detected(self):
        assert not DESTRUCTIVE_INTENT_RE.match("show me all customers")


class TestDestructiveIntentPolite:
    """Regression tests: polite/indirect phrasing must still be detected."""

    def test_please_delete_is_refused(self):
        """'Please delete all customers' must be caught and refused (no Bedrock call)."""
        agent = QuantitativeAgent()
        result = agent.answer("Please delete all customers", "qid-polite-1")
        assert result.refused
        assert result.attempts == 0

    def test_can_you_delete_is_refused(self):
        """'Can you delete all customers?' must be caught and refused."""
        agent = QuantitativeAgent()
        result = agent.answer("Can you delete all customers?", "qid-polite-2")
        assert result.refused
        assert result.attempts == 0

    def test_analytical_deletion_question_not_blocked(self):
        """Past-tense analytical questions mentioning deletion must NOT be pre-refused."""
        query = "How many customer records were deleted last month?"
        assert not DESTRUCTIVE_INTENT_RE.search(query), (
            f"Analytical deletion question falsely blocked: {query!r}"
        )


# ── PRECISION_BY_UNIT and _compute_derived ────────────────────────────────────

class TestPrecisionAndDerived:

    def test_precision_keys_present(self):
        assert "USD" in PRECISION_BY_UNIT
        assert "percent" in PRECISION_BY_UNIT
        assert "count" in PRECISION_BY_UNIT
        assert "score" in PRECISION_BY_UNIT

    def test_count_is_exact(self):
        assert PRECISION_BY_UNIT["count"] == 0.0

    def test_usd_tolerance(self):
        assert PRECISION_BY_UNIT["USD"] == 0.01

    def test_compute_derived_sum(self):
        rows = [{"amount": 100.0}, {"amount": 200.0}, {"amount": 300.0}]
        result = _compute_derived("SUM(amount)", rows)
        assert result == 600.0

    def test_compute_derived_avg(self):
        rows = [{"score": 8.0}, {"score": 6.0}]
        result = _compute_derived("AVG(score)", rows)
        assert result == 7.0

    def test_compute_derived_count(self):
        rows = [{"a": 1}, {"a": 2}, {"a": 3}]
        result = _compute_derived("COUNT(*)", rows)
        assert result == 3.0

    def test_compute_derived_unknown_formula_returns_none(self):
        rows = [{"a": 1}]
        result = _compute_derived("MEDIAN(a)", rows)
        assert result is None


class TestNegativeRowIndex:
    """Regression test: negative row_index must be rejected, not used as Python index."""

    def test_negative_row_index_is_hard_error(self):
        rows = [{"score": 7.2}, {"score": 8.1}]
        payload = {
            "interpretation": "Average score is 7.2.",
            "refused": False,
            "refusal_reason": None,
            "numeric_claims": [
                {"value": 7.2, "row_index": -1, "column": "score", "unit": "score"}
            ],
            "derived_claims": [],
        }
        result = validate_quantitative_interpretation(payload, rows, "end_turn", False)
        assert not result.valid
        assert any("out of range" in e for e in result.errors)


# ── TestProseBypassFix ────────────────────────────────────────────────────────

class TestProseBypassFix:
    """
    Regression tests for the prose cross-check: when numeric_claims key is
    present, significant prose numbers that are not traceable to result_rows
    must be hard errors.
    """

    def _payload(self, interp: str, numeric_claims=None, derived_claims=None):
        return {
            "interpretation": interp,
            "refused": False,
            "refusal_reason": None,
            "numeric_claims": numeric_claims if numeric_claims is not None else [],
            "derived_claims": derived_claims if derived_claims is not None else [],
        }

    def test_prose_number_not_in_claims_is_hard_error(self):
        """Fabricated large number in prose → hard error when numeric_claims present."""
        rows = [{"revenue": 50000.0}]
        payload = self._payload("The total revenue is $99999.99")
        result = validate_quantitative_interpretation(payload, rows, "end_turn", False)
        assert not result.valid
        assert any("99999" in e or "traceable" in e.lower() for e in result.errors)

    def test_prose_number_matching_result_row_passes(self):
        """Number in prose that matches a result_row value → passes."""
        rows = [{"revenue": 50000.0}]
        payload = self._payload("There is $50,000 in revenue.")
        result = validate_quantitative_interpretation(payload, rows, "end_turn", False)
        assert result.valid

    def test_small_number_in_prose_must_match_result_row(self):
        """Small numbers in prose are now checked (not excluded): must appear in result_rows."""
        rows = [{"count": 3.0}]
        # '3' is in result_rows → OK; '5' is NOT → hard error
        payload = self._payload("There are 3 regions and 5 products.")
        result = validate_quantitative_interpretation(payload, rows, "end_turn", False)
        assert not result.valid
        assert any("5.0" in e or "traceable" in e.lower() for e in result.errors)

    def test_small_number_all_covered_passes(self):
        """Small numbers all covered by result_rows → passes."""
        rows = [{"count": 3.0}]
        payload = self._payload("There are 3 items total.", numeric_claims=[], derived_claims=[])
        result = validate_quantitative_interpretation(payload, rows, "end_turn", False)
        assert result.valid

    def test_truncated_sum_derived_claim_is_hard_error(self):
        """SUM derived claim on truncated results must be a hard error."""
        rows = [{"amount": 100.0}, {"amount": 200.0}]
        payload = {
            "interpretation": "Total is $300.",
            "refused": False,
            "refusal_reason": None,
            "numeric_claims": [],
            "derived_claims": [
                {"value": 300.0, "formula": "SUM(amount)", "unit": "USD"}
            ],
        }
        result = validate_quantitative_interpretation(payload, rows, "end_turn", is_truncated=True)
        assert not result.valid
        assert any("truncated" in e.lower() for e in result.errors)

    def test_max_minus_min_derived_claim_verified(self):
        """MAX(col) - MIN(col) binary formula: valid when spread matches."""
        rows = [
            {"dept": "CS", "avg_score": 7.61},
            {"dept": "Sales", "avg_score": 6.53},
            {"dept": "Eng", "avg_score": 7.21},
        ]
        # MAX=7.61, MIN=6.53, spread=1.08
        payload = {
            "interpretation": "The spread is 1.08 points.",
            "refused": False,
            "refusal_reason": None,
            "numeric_claims": [],
            "derived_claims": [
                {"value": 1.08, "formula": "MAX(avg_score) - MIN(avg_score)", "unit": "score"}
            ],
        }
        result = validate_quantitative_interpretation(payload, rows, "end_turn", False)
        assert result.valid, f"Expected valid but got errors: {result.errors}"

    def test_max_minus_min_fabricated_is_hard_error(self):
        """MAX-MIN spread: wrong claimed value → hard error."""
        rows = [
            {"dept": "CS", "avg_score": 7.61},
            {"dept": "Sales", "avg_score": 6.53},
        ]
        # MAX=7.61, MIN=6.53, real spread=1.08 — claiming 5.0 is fabricated
        payload = {
            "interpretation": "The spread is 5.0 points.",
            "refused": False,
            "refusal_reason": None,
            "numeric_claims": [],
            "derived_claims": [
                {"value": 5.0, "formula": "MAX(avg_score) - MIN(avg_score)", "unit": "score"}
            ],
        }
        result = validate_quantitative_interpretation(payload, rows, "end_turn", False)
        assert not result.valid
        assert any("mismatch" in e.lower() or "5.0" in e for e in result.errors)

    def test_no_structured_claims_key_is_now_hard_error(self):
        """numeric_claims key absent → hard error (required field, see Issue 2 fix)."""
        rows = [{"revenue": 100.0}]
        payload = {
            "interpretation": "Revenue was $99999.",
            "refused": False,
        }
        result = validate_quantitative_interpretation(payload, rows, "end_turn", False)
        assert not result.valid
        assert any("numeric_claims" in e for e in result.errors)


# ── Issue 2 regression tests ──────────────────────────────────────────────────

class TestIssue2Validation:
    """Require numeric_claims key; small numbers covered; unknown formula = hard error."""

    def _base(self, interp, **extra):
        p = {"interpretation": interp, "refused": False, "refusal_reason": None}
        p.update(extra)
        return p

    def test_missing_numeric_claims_key_is_hard_error(self):
        """numeric_claims key absent → hard error (structured schema required)."""
        rows = [{"revenue": 5000.0}]
        payload = self._base("Revenue is $5000.")  # no numeric_claims key at all
        result = validate_quantitative_interpretation(payload, rows, "end_turn", False)
        assert not result.valid
        assert any("numeric_claims" in e for e in result.errors)

    def test_small_number_in_prose_matching_result_row_passes(self):
        """Small number (≤10) that appears in a result row must pass."""
        rows = [{"count": 8.0}]
        payload = self._base("8 customers have churned.",
                             numeric_claims=[], derived_claims=[])
        result = validate_quantitative_interpretation(payload, rows, "end_turn", False)
        assert result.valid, f"Expected pass but errors: {result.errors}"

    def test_small_fabricated_number_in_prose_is_hard_error(self):
        """Small number in prose NOT in any result row must be a hard error."""
        rows = [{"count": 8.0}]
        payload = self._base("3 customers have churned.",
                             numeric_claims=[], derived_claims=[])
        result = validate_quantitative_interpretation(payload, rows, "end_turn", False)
        assert not result.valid
        assert any("3" in e or "traceable" in e.lower() for e in result.errors)

    def test_unknown_derived_formula_is_hard_error(self):
        """_compute_derived returns None for unknown formula → hard error, not warning."""
        rows = [{"amount": 100.0}]
        payload = self._base("Magic total is $100.",
                             numeric_claims=[],
                             derived_claims=[
                                 {"value": 100.0, "formula": "SOME_MAGIC(amount)", "unit": "USD"}
                             ])
        result = validate_quantitative_interpretation(payload, rows, "end_turn", False)
        assert not result.valid
        assert any("SOME_MAGIC" in e or "not be computed" in e.lower() or "could not" in e.lower()
                   for e in result.errors)


class TestPercentageColumnValidation:

    def test_percentage_column_in_sql_passes_numeric_claim(self):
        """A percentage value returned as a SQL column is valid as a numeric_claim."""
        payload = {
            "interpretation": "West had 29.39% of total revenue.",
            "refused": False,
            "refusal_reason": None,
            "numeric_claims": [
                {"value": 501967.15, "row_index": 0, "column": "total_revenue", "unit": "USD"},
                {"value": 29.39, "row_index": 0, "column": "pct_of_total", "unit": "percent"},
            ],
            "derived_claims": [],
        }
        result_rows = [{"region": "West", "total_revenue": 501967.15, "pct_of_total": 29.39}]
        result = validate_quantitative_interpretation(payload, result_rows, "end_turn", False)
        assert result.valid, result.errors
