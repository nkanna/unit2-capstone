"""
Regression tests for utils.parse_json_object.

Fenced SQL responses taken verbatim from evidence/live_calls.jsonl to
reproduce the actual parsing failure that caused Q2/Q3/Q4 queries to fail
before this fix was introduced.
"""
import pytest
from utils import parse_json_object


# ── Actual fenced responses from evidence/live_calls.jsonl ────────────────────

FENCED_SQL_Q2 = (
    "```json\n"
    "{\n"
    '  "sql": "SELECT region, ROUND(COALESCE(SUM(amount), 0), 2) AS total_revenue '
    "FROM sales WHERE strftime('%Y', sale_date) = '2024' GROUP BY region "
    'ORDER BY total_revenue DESC",\n'
    '  "reasoning": "This query sums all sales amounts grouped by region for the year 2024."\n'
    "}\n"
    "```"
)

FENCED_SQL_Q4 = (
    "```json\n"
    "{\n"
    '  "sql": "SELECT department, ROUND(AVG(satisfaction_score), 2) AS avg_satisfaction '
    "FROM employees WHERE status = 'active' GROUP BY department "
    'ORDER BY avg_satisfaction ASC LIMIT 1",\n'
    '  "reasoning": "Calculates average satisfaction score per active department."\n'
    "}\n"
    "```"
)

# Q3 qualitative_generation call_index=1: prose response that must be rejected
PROSE_RESPONSE = (
    "Based on the retrieved context from the Employee Satisfaction Policy (HR-005), "
    "here is what it says about manager accountability:\n\n"
    "1. **Performance Reviews:** Manager effectiveness scores from direct reports "
    "are included in manager performance reviews."
)


class TestParseJsonObject:

    def test_fenced_sql_q2_parses(self):
        """Actual Q2 sql_generation fenced response must parse correctly."""
        result = parse_json_object(FENCED_SQL_Q2)
        assert "sql" in result
        assert "SELECT" in result["sql"]
        assert "reasoning" in result

    def test_fenced_sql_q4_parses(self):
        """Actual Q4 sql_generation fenced response must parse correctly."""
        result = parse_json_object(FENCED_SQL_Q4)
        assert "sql" in result
        assert "avg_satisfaction" in result["sql"]

    def test_plain_json_parses(self):
        """Plain JSON without fences parses correctly."""
        result = parse_json_object('{"route": "qualitative", "reasoning": "policy question"}')
        assert result["route"] == "qualitative"

    def test_plain_json_with_whitespace_parses(self):
        """Plain JSON with leading/trailing whitespace parses correctly."""
        result = parse_json_object('  \n{"key": "value"}\n  ')
        assert result["key"] == "value"

    def test_prose_response_raises(self):
        """Prose response (not JSON and not fenced) must raise ValueError."""
        with pytest.raises(ValueError, match="not a JSON object"):
            parse_json_object(PROSE_RESPONSE)

    def test_empty_string_raises(self):
        """Empty string must raise ValueError."""
        with pytest.raises(ValueError):
            parse_json_object("")

    def test_whitespace_only_raises(self):
        """Whitespace-only string must raise ValueError."""
        with pytest.raises(ValueError):
            parse_json_object("   \n  ")

    def test_malformed_json_in_fence_raises(self):
        """Malformed JSON inside a fence must raise ValueError."""
        with pytest.raises(ValueError, match="Malformed JSON"):
            parse_json_object("```json\n{bad json here\n```")

    def test_json_list_raises(self):
        """A top-level JSON array is not a JSON object — must raise."""
        with pytest.raises(ValueError):
            parse_json_object("[1, 2, 3]")

    def test_fenced_json_list_raises(self):
        """A fenced top-level JSON array is not a JSON object — must raise."""
        with pytest.raises(ValueError, match="Expected a JSON object"):
            parse_json_object("```json\n[1, 2, 3]\n```")

    def test_multiple_fenced_blocks_raises(self):
        """Multiple fenced blocks must raise ValueError (malformed inner JSON)."""
        two_blocks = "```json\n{}\n```\n```json\n{}\n```"
        with pytest.raises(ValueError):
            parse_json_object(two_blocks)

    def test_plain_json_list_raises(self):
        """A top-level JSON array without fences must raise ValueError."""
        with pytest.raises(ValueError):
            parse_json_object('["a", "b"]')
