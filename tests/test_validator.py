"""
Offline unit tests for validation/validator.py.

All tests run without network access, without a database, and without
calling Claude.  No assertions are fabricated — every test either passes
or the assertion error shows what actually happened.
"""

import pytest
from validation.validator import ValidationResult, validate_qualitative_response, validate_sql


# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_IDS = {"security_policy.md::chunk_1", "security_policy.md::chunk_2"}

def _good_payload(refused=False):
    return {
        "answer": "Passwords must be at least 16 characters.",
        "citations": [
            {
                "chunk_id": "security_policy.md::chunk_1",
                "source": "security_policy.md",
                "excerpt": "Passwords must be at least 16 characters",
            }
        ],
        "refused": refused,
        "refusal_reason": None,
    }


# ── validate_qualitative_response ─────────────────────────────────────────────

class TestQualitativeValidation:

    def test_valid_response_passes(self):
        result = validate_qualitative_response(_good_payload(), VALID_IDS, "end_turn")
        assert result.valid
        assert result.errors == []

    def test_fabricated_citation_fails(self):
        """A chunk_id not in the retrieved set must be rejected."""
        payload = _good_payload()
        payload["citations"][0]["chunk_id"] = "made_up_file.md::chunk_99"
        result = validate_qualitative_response(payload, VALID_IDS, "end_turn")
        assert not result.valid
        assert any("made_up_file.md::chunk_99" in e for e in result.errors)

    def test_missing_answer_field_fails(self):
        payload = {"citations": [], "refused": False}
        result = validate_qualitative_response(payload, VALID_IDS, "end_turn")
        assert not result.valid
        assert any("answer" in e for e in result.errors)

    def test_missing_citations_field_fails(self):
        payload = {"answer": "Some text", "refused": False}
        result = validate_qualitative_response(payload, VALID_IDS, "end_turn")
        assert not result.valid
        assert any("citations" in e for e in result.errors)

    def test_missing_refused_field_fails(self):
        payload = {"answer": "Some text", "citations": []}
        result = validate_qualitative_response(payload, VALID_IDS, "end_turn")
        assert not result.valid
        assert any("refused" in e for e in result.errors)

    def test_truncated_response_fails(self):
        """stop_reason != 'end_turn' means the response was cut off."""
        payload = _good_payload()
        result = validate_qualitative_response(payload, VALID_IDS, "max_tokens")
        assert not result.valid
        assert any("truncated" in e.lower() for e in result.errors)

    def test_wrong_type_answer_fails(self):
        payload = _good_payload()
        payload["answer"] = 42
        result = validate_qualitative_response(payload, VALID_IDS, "end_turn")
        assert not result.valid
        assert any("answer" in e for e in result.errors)

    def test_wrong_type_citations_fails(self):
        payload = _good_payload()
        payload["citations"] = "not a list"
        result = validate_qualitative_response(payload, VALID_IDS, "end_turn")
        assert not result.valid
        assert any("citations" in e for e in result.errors)

    def test_wrong_type_refused_fails(self):
        payload = _good_payload()
        payload["refused"] = "yes"
        result = validate_qualitative_response(payload, VALID_IDS, "end_turn")
        assert not result.valid
        assert any("refused" in e for e in result.errors)

    def test_refused_with_empty_answer_passes(self):
        """A refused response with an empty answer is valid."""
        payload = {
            "answer": "",
            "citations": [],
            "refused": True,
            "refusal_reason": "No relevant context found.",
        }
        result = validate_qualitative_response(payload, VALID_IDS, "end_turn")
        assert result.valid
        assert result.errors == []

    def test_refused_with_non_empty_answer_warns(self):
        """refused=True + non-empty answer is suspicious — should warn."""
        payload = {
            "answer": "Here is an answer anyway.",
            "citations": [],
            "refused": True,
            "refusal_reason": "Context insufficient.",
        }
        result = validate_qualitative_response(payload, VALID_IDS, "end_turn")
        # Should still be valid (warnings don't block) but must have a warning
        assert result.valid
        assert any("refused=true" in w.lower() for w in result.warnings)

    def test_empty_payload_fails(self):
        result = validate_qualitative_response({}, VALID_IDS, "end_turn")
        assert not result.valid

    def test_citation_missing_chunk_id_fails(self):
        payload = _good_payload()
        payload["citations"][0] = {"source": "security_policy.md", "excerpt": "..."}
        result = validate_qualitative_response(payload, VALID_IDS, "end_turn")
        assert not result.valid
        assert any("chunk_id" in e for e in result.errors)

    def test_citation_without_excerpt_warns(self):
        payload = _good_payload()
        del payload["citations"][0]["excerpt"]
        result = validate_qualitative_response(payload, VALID_IDS, "end_turn")
        assert result.valid   # missing excerpt is a warning, not an error
        assert any("excerpt" in w for w in result.warnings)

    def test_multiple_valid_citations_pass(self):
        payload = _good_payload()
        payload["citations"].append({
            "chunk_id": "security_policy.md::chunk_2",
            "source": "security_policy.md",
            "excerpt": "another quote",
        })
        result = validate_qualitative_response(payload, VALID_IDS, "end_turn")
        assert result.valid


# ── validate_sql ──────────────────────────────────────────────────────────────

class TestSqlValidation:

    def test_plain_select_passes(self):
        result = validate_sql("SELECT COUNT(*) FROM customers")
        assert result.valid

    def test_select_with_where_passes(self):
        result = validate_sql(
            "SELECT name, region FROM customers WHERE status = 'active'"
        )
        assert result.valid

    def test_empty_sql_fails(self):
        result = validate_sql("")
        assert not result.valid

    def test_drop_table_fails(self):
        result = validate_sql("DROP TABLE customers")
        assert not result.valid
        assert any("DROP" in e.upper() or "SELECT" in e.upper() for e in result.errors)

    def test_delete_fails(self):
        result = validate_sql("DELETE FROM customers WHERE id = 1")
        assert not result.valid

    def test_insert_fails(self):
        result = validate_sql("INSERT INTO customers VALUES (1, 'x')")
        assert not result.valid

    def test_update_fails(self):
        result = validate_sql("UPDATE employees SET salary = 0")
        assert not result.valid

    def test_line_comment_injection_fails(self):
        result = validate_sql("SELECT * FROM customers -- DROP TABLE customers")
        assert not result.valid

    def test_block_comment_injection_fails(self):
        result = validate_sql("SELECT * FROM customers /* injected */")
        assert not result.valid

    def test_statement_chaining_fails(self):
        """Semicolon followed by another statement is a chaining attack."""
        result = validate_sql("SELECT 1; DROP TABLE customers")
        assert not result.valid

    def test_select_star_warns(self):
        result = validate_sql("SELECT * FROM employees")
        assert result.valid           # warning only, not an error
        assert any("SELECT *" in w for w in result.warnings)

    def test_pragma_fails(self):
        result = validate_sql("PRAGMA table_info(customers)")
        assert not result.valid

    def test_attach_fails(self):
        result = validate_sql("ATTACH DATABASE '/tmp/evil.db' AS evil")
        assert not result.valid


# ── Issue 4 regression tests ──────────────────────────────────────────────────

class TestIssue4QualitativeValidation:

    def test_non_refused_answer_with_no_citations_fails(self):
        """Non-refused answer with empty citations must fail validation."""
        payload = {
            "answer": "The policy says passwords must be 16 characters.",
            "citations": [],
            "refused": False,
            "refusal_reason": None,
        }
        result = validate_qualitative_response(payload, VALID_IDS, "end_turn")
        assert not result.valid
        assert any("citation" in e.lower() for e in result.errors)

    def test_refused_answer_with_no_citations_passes(self):
        """Refused answers do not require citations."""
        payload = {
            "answer": "",
            "citations": [],
            "refused": True,
            "refusal_reason": "No relevant context found.",
        }
        result = validate_qualitative_response(payload, VALID_IDS, "end_turn")
        assert result.valid

    def test_excerpt_not_in_chunk_text_warns(self):
        """Excerpt not found in chunk text → hard error (not just a warning)."""
        payload = {
            "answer": "Some answer.",
            "citations": [{
                "chunk_id": "security_policy.md::chunk_1",
                "source": "security_policy.md",
                "excerpt": "This text is completely fabricated and not real.",
            }],
            "refused": False,
            "refusal_reason": None,
        }
        chunk_texts = {
            "security_policy.md::chunk_1": "Actual chunk content here about password policy."
        }
        result = validate_qualitative_response(
            payload, VALID_IDS, "end_turn", chunk_texts=chunk_texts
        )
        assert not result.valid, "Expected invalid — excerpt mismatch must be a hard error"
        assert any("excerpt" in e.lower() for e in result.errors)
        assert not any("excerpt" in w.lower() for w in result.warnings)

    def test_excerpt_mismatch_is_hard_error_not_warning(self):
        """Fabricated excerpt must produce a hard error, not a warning."""
        payload = _good_payload()
        chunk_texts = {"security_policy.md::chunk_1": "Actual chunk content here."}
        result = validate_qualitative_response(payload, VALID_IDS, "end_turn",
                                               chunk_texts=chunk_texts)
        assert not result.valid
        assert any("excerpt" in e.lower() for e in result.errors)
        assert not any("excerpt" in w.lower() for w in result.warnings)

    def test_excerpt_whitespace_normalized_match_passes(self):
        """Extra whitespace in excerpt must not cause a false mismatch."""
        payload = _good_payload()
        payload["citations"][0]["excerpt"] = "Passwords  must  be at  least   16 characters"
        chunk_texts = {
            "security_policy.md::chunk_1":
            "Some preamble. Passwords must be at least 16 characters in length."
        }
        result = validate_qualitative_response(payload, VALID_IDS, "end_turn",
                                               chunk_texts=chunk_texts)
        assert result.valid
        assert not any("excerpt" in e.lower() for e in result.errors)

    def test_excerpt_found_in_chunk_text_passes_without_warning(self):
        """Excerpt that IS in chunk text → passes with no excerpt warning."""
        payload = {
            "answer": "Some answer.",
            "citations": [{
                "chunk_id": "security_policy.md::chunk_1",
                "source": "security_policy.md",
                "excerpt": "password policy",
            }],
            "refused": False,
            "refusal_reason": None,
        }
        chunk_texts = {
            "security_policy.md::chunk_1": "Actual chunk content about password policy here."
        }
        result = validate_qualitative_response(
            payload, VALID_IDS, "end_turn", chunk_texts=chunk_texts
        )
        assert result.valid
        assert not any("not found in chunk" in w.lower() for w in result.warnings)
