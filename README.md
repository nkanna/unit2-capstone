# Unit 2 Capstone: Multi-Agent RAG System with Claude

A Python 3.11 CLI application that answers questions about company policies
(qualitative) and business data (quantitative) using a multi-agent architecture.
All data is entirely fictional and created for educational purposes only.

---

## Corrections

The following values were corrected from earlier drafts to match actual files:

- **Test count**: 165 total — test_manager.py (28), test_qualitative.py (9), test_quantitative.py (82), test_validator.py (34), test_parse_json.py (12).
- **Department employee counts**: HR and Sales each corrected from earlier estimates to 8 active employees each, matching `SELECT COUNT(*) … GROUP BY department WHERE status='active'` on the actual seed database.
- **Token totals**: 169,313 input / 43,503 output / $1.276532 across 56 query summaries, summed from the actual `tokenomics_log.jsonl`.
- **Code-fence parsing failure**: Q2, Q3, and Q4 initial failures were caused by SQL generation responses arriving as JSON inside markdown code fences (` ```json\n{...}\n``` `). `json.loads()` cannot parse fenced input; `utils.parse_json_object()` was added to strip a single fence before parsing. The original failures are preserved in `evidence/live_calls.jsonl`.
- **Trust-but-Verify examples**: built from saved evidence files (actual Bedrock responses). Removed a fabricated claim ("Stale review invalidation") that does not appear in `code_review_policy.md`. Example 3 replaced with the satisfaction follow-up query, including a confirmed human reviewer correction.
- **Validation design**: corrected several descriptions to reflect fixes applied after earlier drafts (see Validation Design section).
- **`.env.example`**: variable names corrected from `COST_PER_1K_*` to `COST_PER_M_*` to match what `tokenomics/logger.py` actually reads.

---

## Architecture

```
                              ┌─────────────────┐
                              │  Validator       │
                              │  (all responses) │
                              └────────┬────────┘
                                       │ validates every Claude call
┌──────┐   ┌──────────┐   ┌──────────▼───────────────────────────┐
│ User │──▶│ main.py  │──▶│              Manager                  │
│ CLI  │◀──│          │◀──│  classify → route → synthesize        │
└──────┘   └──────────┘   └──────┬────────────────────┬──────────┘
                                  │                    │
               ┌──────────────────▼──┐     ┌──────────▼──────────────┐
               │  QualitativeAgent   │     │  QuantitativeAgent      │
               │  ChromaDB retrieval │     │  SQL generation          │
               │  + Claude generation│     │  + SQLite execution      │
               └──────────┬──────────┘     └──────────┬──────────────┘
                          │                           │
               ┌──────────▼──────────┐     ┌──────────▼──────────────┐
               │     ChromaDB        │     │   SQLite (read-only)    │
               │  (cosine, 39 chunks)│     │   customers/sales/emps  │
               └─────────────────────┘     └─────────────────────────┘
                          │                           │
                          └───────────┬───────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │   AWS Bedrock            │
                         │   us.anthropic.claude-   │
                         │   sonnet-4-6             │
                         └─────────────────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │   TokenomicsLogger       │
                         │   tokenomics_log.jsonl   │
                         └─────────────────────────┘
```

---

## Dataset (Fictional — Educational Use Only)

All data represents the entirely fictional company "Acme Corp". No real persons,
companies, or events are depicted.

### Policy documents (`data/documents/`)

| File | Policy ID | Topic |
|------|-----------|-------|
| `security_policy.md` | SEC-001 | Passwords, MFA, data classification, incident response |
| `code_review_policy.md` | ENG-002 | PR size, approvals, review checklist, automated checks |
| `customer_complaints_policy.md` | CS-003 | Intake, SLAs, escalation procedures |
| `customer_success_policy.md` | CS-004 | Segmentation, onboarding, health scoring, QBRs |
| `employee_satisfaction_policy.md` | HR-005 | Pulse surveys, eNPS, manager accountability |

### SQLite database (`data/database.sqlite`)

| Table | Rows | Key columns |
|-------|------|-------------|
| `customers` | 50 | id, name, region, tier, status, signup_date, mrr, industry |
| `sales` | 375 | id, customer_id, amount, sale_date, region, rep_name, product |
| `employees` | 40 | id, name, department, region, hire_date, annual_salary, satisfaction_score, status |

Seed script: `data/seed_database.py` (fixed random seed 42, idempotent)

---

## AWS Bedrock Setup

This project uses AWS Bedrock instead of a direct Anthropic API key.

- **Model**: `us.anthropic.claude-sonnet-4-6`
- **Region**: `us-east-1`

Configure credentials with:
```bash
aws configure
```

Or set environment variables `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_DEFAULT_REGION`. Never paste credentials into project files.

See `.env.example` for all supported environment variables. Copy it to `.env`
(already excluded from git by `.gitignore`).

---

## Quick Start (Fresh Clone)

Complete setup from a fresh clone:

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure AWS Bedrock (no credentials in this repo)
aws configure                  # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION

# 4. Copy and review environment variables (optional overrides)
cp .env.example .env

# 5. Seed the SQLite database
python data/seed_database.py

# 6. Build Chroma vector storage (chroma_store/ is gitignored — must rebuild after clone)
python ingest.py --reset

# 7. Run the CLI
python main.py

# 8. Run tests (all offline — no AWS calls required)
pytest tests/ -v
```

---

## Installation

Python 3.11 required.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data Seeding

```bash
python data/seed_database.py                    # default: data/database.sqlite
python data/seed_database.py --db /tmp/test.db  # alternate path for testing
```

Creates the target file with 50 customers, 375 sales, 40 employees.
The script is idempotent (drop-and-recreate). Fixed random seed 42 ensures
identical output on every run.

---

## Document Ingestion

```bash
python ingest.py                                          # default: chroma_store/
python ingest.py --reset                                  # delete and rebuild
python ingest.py --chroma-path /tmp/test_chroma --reset  # alternate path for testing
```

Chunks the 5 policy documents and stores them in the target ChromaDB directory
(cosine similarity, 39 chunks total).

---

## Running the CLI

**Interactive mode:**
```bash
python main.py
```
Type `exit` or `quit` to stop. Type `reset` to clear conversation history.

**Single-query mode:**
```bash
python main.py "What is the password policy?"
```

---

## Running Tests

```bash
pytest tests/ -v
```

165 tests, all offline (no Bedrock or ChromaDB calls required):
- `test_validator.py` — 34 tests (SQL validation, qualitative response validation, excerpt matching)
- `test_qualitative.py` — 9 tests (retrieval pre-filter, citation validation, retries)
- `test_quantitative.py` — 82 tests (SQL safety, structured numeric claims, destructive intent, prose bypass)
- `test_manager.py` — 28 tests (routing, history, synthesis grounding, refusal propagation, error handling)
- `test_parse_json.py` — 12 tests (code-fence stripping, plain JSON, rejection cases using actual evidence payloads)

---

## Supported Questions

### Qualitative (policy/documentation)
- "What is the security policy for passwords?"
- "What are the PR approval criteria?"
- "How are customer complaints escalated?"
- "What is the employee onboarding process?"

### Quantitative (data/metrics)
- "What was total revenue by region in 2024?"
- "How many employees are in each department?"
- "What is the customer churn rate?"
- "What is the average employee satisfaction score by department?"

### Combined (both sources)
- "What does the customer success policy say about at-risk customers, and how many customers have churned?"
- "What is the remote work policy and how many employees are currently active?"

### Out of scope (refused)
- General world knowledge ("What is the capital of France?")
- Requests to modify data ("Delete all customer records")

---

## Metric Definitions

These definitions are built into the SQL-generation prompt to ensure consistent
query construction:

| Metric | Definition |
|--------|-----------|
| Revenue | `SUM(sales.amount)` for a period or grouping |
| Churn rate (%) | `COUNT(status='churned') * 100.0 / COUNT(*) FROM customers` |
| Active customers | `WHERE status = 'active'` |
| MRR | `customers.mrr` column (monthly recurring revenue per customer) |
| Employee satisfaction | `AVG(satisfaction_score) WHERE status='active'` (scale 1–10) |
| Average salary | `AVG(annual_salary)` in USD per year |

---

## Conversation History

The Manager maintains a bounded session history of the last 10 turns
(`MAX_HISTORY_TURNS = 10` in `agents/manager.py`). History is passed as
context to all agents on each call so follow-up questions can refer to
earlier answers.

- History persists for the lifetime of one `python main.py` session.
- Type `reset` in the CLI to clear history.
- History contains only validated, non-refused answers. Failed responses
  (classification errors, unsupported queries) are not added.
- When a SQL generation response arrives as a JSON code-fenced block
  (` ```json\n{...}\n``` `), `utils.parse_json_object()` strips the fence
  before parsing. Earlier evidence runs showed this as a parsing failure;
  it is now corrected. Type `reset` to clear session history between
  independent questions.

---

## Validation Design

Every Claude response is validated before use. Validation failures trigger
a retry (up to 2 total calls per step); if retries are exhausted the query
is refused rather than returning an unvalidated answer.

### Qualitative responses (`validate_qualitative_response`)
- `stop_reason` must be `"end_turn"` (not truncated)
- Required fields: `answer`, `citations`, `refused`
- Non-refused answers must include at least one citation
- Each citation `chunk_id` must be present in the retrieved set
- Each citation `excerpt` is checked as a substring of the retrieved chunk text
  (whitespace-normalised). An excerpt not found in its chunk is a hard error.

**Known limitation**: Excerpt substring matching does not prove every claim in the
answer is supported — it only confirms the quoted text appeared in the retrieved chunk.

### SQL validation (`validate_sql`) — three layers
1. **Regex pre-filter**: blocks DML keywords, comment markers
2. **Quote-aware statement counter**: rejects multi-statement payloads
3. **SQLite authorizer on in-memory schema**: catches unknown tables, columns,
   and functions during SQL PREPARE (before any data is touched)

Additionally: all SQL is executed on a read-only URI connection
(`?mode=ro`) with a runtime authorizer as defense-in-depth.

### Interpretation responses (`validate_quantitative_interpretation`)
- `stop_reason` check
- Structured `numeric_claims` with `{value, row_index, column, unit}`:
  each claim is validated against `result_rows[row_index][column]` using
  metric-appropriate precision:
  - USD: ±$0.01  |  percent: ±0.1  |  count: exact  |  score: ±0.05
- `derived_claims` with `{value, formula, unit}` computed in Python.
  Supported single-aggregate forms: `SUM(col)`, `AVG(col)`, `COUNT(*)`,
  `MAX(col)`, `MIN(col)`. Supported binary form: `MAX(col) - MIN(col)`
  (and `+` variant). Unknown formulas are hard errors (trigger retry).
- `numeric_claims` key is **required** in every interpretation response;
  its absence is a hard error.
- **Prose cross-check** (when `numeric_claims` key is present): any
  number ≥ 1.0 in the interpretation text must match a value in
  `result_rows` or an explicit claim. Unmatched prose numbers are hard
  errors that trigger retry.
- **Truncation guard**: `SUM`, `AVG`, and `COUNT` `derived_claims` are
  rejected with a hard error when results are truncated — computing them
  on a partial result set would give wrong totals. `MAX`/`MIN` on
  truncated results generate warnings only (sort order unknown).

**Known limitation**: Passing the numeric consistency check does not prove
the full interpretation is correct. It confirms cited values appeared in the
result set, not that they are interpreted correctly.

**Known limitation**: Prose cross-check uses 2% relative tolerance for
matching. A semantically wrong value that happens to appear in result_rows
will pass this check.

### Classification responses (`validate_classification_response`)
- Route must be one of: `qualitative`, `quantitative`, `both`, `unsupported`
- `both` route requires non-empty subqueries for both agents

### Synthesis responses (`validate_synthesis_response`)
- Required fields: `synthesis`, `refused`
- When quantitative result rows and/or retrieved policy texts are provided,
  every significant prose number (≥ 1.0, excluding calendar years) must
  either match a value in `result_rows` (±2% relative) or appear as a
  token in the retrieved policy chunk texts. Numbers traceable to neither
  source are **hard errors** that trigger a synthesis retry.
- If all synthesis retries fail, the fallback (separate qual + quant
  sections concatenated) is used and stored in history instead.
- **Known limitation**: Policy-number matching is substring-based; a number
  that happens to appear in policy text is considered grounded even if the
  surrounding context differs.

### Destructive intent pre-filter
Before any SQL generation, `DESTRUCTIVE_INTENT_RE` checks for keywords like
`delete`, `drop`, `truncate`, `update all`. Matched queries are refused
immediately without calling Bedrock or touching the database.

---

## Tokenomics

- **Input cost**: $3.30 per million tokens (us-east-1, us.anthropic.claude-sonnet-4-6)
- **Output cost**: $16.50 per million tokens
- **Log**: `tokenomics_log.jsonl` — two record types:
  - `"call"`: one entry per individual Bedrock call
  - `"query_summary"`: one entry per user turn (totals, no double-counting)

**Measured totals across 56 query summaries in this project:**
- Total input tokens: 169,313
- Total output tokens: 43,503
- Total estimated cost: $1.276532

---

## Requirement Checklist

### Bronze

| # | Requirement | Status | Evidence |
|---|------------|--------|---------|
| B1 | `ingest.py` chunks, embeds, stores in ChromaDB | ✓ | `ingest.py`: `chunk_text()`, `load_documents()`, `_sync_collection()` |
| B2 | Manager supports qualitative, quantitative, combined | ✓ | `agents/manager.py:169` `Manager.answer()`, routes qualitative/quantitative/both/unsupported |
| B3 | Qualitative: grounded answers, citations, refusal when unavailable | ✓ | `agents/qualitative.py`: RETRIEVAL_THRESHOLD pre-filter, citation validation, refused=True on miss |
| B4 | Quantitative agent generates SQL through Claude | ✓ | `agents/quantitative.py:273` SQL_GENERATION_SYSTEM prompt, `bc.invoke()` call |
| B5 | `validate_sql` blocks non-SELECT and unsafe queries | ✓ | `validation/validator.py:186` three-layer validation; 13 SQL tests in `test_validator.py` |
| B6 | Every Claude response passes through validation | ✓ | manager.py:204, quantitative.py:297-418, qualitative.py: all `bc.invoke` calls followed by validator |
| B7 | Tokenomics logs actual tokens and estimated cost | ✓ | `tokenomics/logger.py`: `log_call()`, `log_query_summary()`; `tokenomics_log.jsonl` |
| B8 | `main.py` provides interactive CLI | ✓ | `main.py`: interactive loop, exit/quit/reset, single-query mode |
| B9 | README with architecture, setup, usage | ✓ | This document |
| B10 | README has Trust-but-Verify section (3 real examples) | ✓ | See Trust-but-Verify section below |
| B11 | `tokenomics_log.jsonl` with ≥10 real user queries | ✓ | 56 `query_summary` records (47 full-application runs through Manager; 9 earlier direct-agent checks from Stages 1–4) |

### Silver

| Requirement | Status | Evidence |
|------------|--------|---------|
| Conversation history for follow-up questions | ✓ | `agents/manager.py:45` `SessionHistory`, `MAX_HISTORY_TURNS=10`, `to_messages()` passed to agents |

### Gold

| Requirement | Status | Evidence |
|------------|--------|---------|
| Meaningful unit tests for manager, qualitative, quantitative, validator | ✓ | 165 tests in `tests/`: test_manager.py (28), test_qualitative.py (9), test_quantitative.py (82), test_validator.py (34), test_parse_json.py (12) |

---

## Trust-but-Verify

The following examples are built from saved evidence files — actual Bedrock
responses captured during live runs. Raw outputs are copied verbatim from the
evidence files. Reviewer corrections are shown separately and clearly labelled.

---

### Trust-but-Verify Example 1 — Clean Pass (Qualitative)

**Query**: "What are the requirements for PR approval in the code review policy?"
**Query ID**: `d977166e1067`
**Route**: qualitative
**Refused**: no

**Output** (actual, from `evidence/query_results.jsonl`):
> According to the code review policy (ENG-002), the requirements for PR approval are as follows:
>
> 1. **Minimum Approvals:** Every PR must have at least two approvals before merging to main or a release branch.
> 2. **Domain Owner Requirement:** At least one reviewer must be a domain owner (maintainer of the affected subsystem).
> 3. **Security-Critical Code:** PRs touching security-critical code (auth, payments, cryptography, secrets handling) require an additional approval from a member of the Security Review Board.
> 4. **Author Restriction:** The author may not approve their own PR.
> 5. **Bot Reviews:** Reviews from automated bots do not count toward the human-reviewer minimum.

**Evidence**:
- Citations: 6 total — 5 from `code_review_policy.md::chunk_2`, 1 from `::chunk_6` (verbatim excerpts, substring-verified)
  - Excerpt 1: `"Every PR must have at least **two approvals** before merging to main or a release branch."` [chunk_2]
  - Excerpt 2: `"At least one reviewer must be a domain owner (maintainer of the affected subsystem)."` [chunk_2]
  - Excerpt 3: `"PRs touching security-critical code (auth, payments, cryptography, secrets handling) require an additional approval from a member of the Security Review Board."` [chunk_2]
  - Excerpt 4: `"The author may not approve their own PR."` [chunk_2]
  - Excerpt 5: `"Reviews from automated bots do not count toward the human-reviewer minimum."` [chunk_2]
  - Excerpt 6: `"Reviewers must respond to re-review requests within **four business hours**."` [chunk_6]
- SQL: N/A
- Result rows: N/A

**Validation outcome**: valid=True, errors=[], warnings=[]
Tokens: 3,046 in / 1,414 out | Cost: $0.033383

**Review decision** (confirmed by project author): ACCEPT
**Rationale**: All five approval-requirement claims match the actual policy rules in `code_review_policy.md` (verified against source). All six excerpts are verbatim quotes confirmed present in their retrieved chunks (substring check passed). No numeric claims. No interpretation layer.

---

### Trust-but-Verify Example 2 — Clean Pass (Quantitative, all numbers verified)

**Query**: "What is the total revenue by region for 2024?"
**Query ID**: `1c4c017c9be7`
**Route**: quantitative
**Refused**: no

**Output** (actual):
> In 2024, total revenue across all regions amounted to approximately $1,713,328.38. The West region led all regions with roughly $501,967.15 in revenue, followed closely by EMEA at about $440,725.04. The East and North regions contributed similarly, at approximately $293,775.07 and $286,255.82 respectively. The South region generated the least revenue at around $190,605.30.

**Evidence**:
- Citations: none (quantitative)
- SQL: `SELECT region, ROUND(COALESCE(SUM(amount), 0), 2) AS total_revenue FROM sales WHERE strftime('%Y', sale_date) = '2024' GROUP BY region ORDER BY total_revenue DESC`
- Result rows: `[{"region": "West", "total_revenue": 501967.15}, {"region": "EMEA", "total_revenue": 440725.04}, {"region": "East", "total_revenue": 293775.07}, {"region": "North", "total_revenue": 286255.82}, {"region": "South", "total_revenue": 190605.3}]`

**Validation outcome**: valid=True, errors=[], warnings=[]
Tokens: 2,404 in / 743 out | Cost: $0.025195

**Review decision** (confirmed by project author): ACCEPT
**Rationale**: All five regional revenue figures independently verified: 501,967.15 + 440,725.04 + 293,775.07 + 286,255.82 + 190,605.30 = 1,713,328.38 ✓. Each figure validated by the system via `numeric_claims`; total via `SUM(total_revenue)` derived claim. Result saved in `evidence/query_results.jsonl` (label `Q2_quantitative_retry`). Initial run (query_id `5cd4a742affb`) failed because the SQL generation response arrived as a JSON code-fenced block that `json.loads()` could not parse; original failure preserved in `evidence/live_calls.jsonl`.

---

### Trust-but-Verify Example 3 — Follow-up Requiring Reviewer Correction

**Query**: "Which department scored lowest, and what does the policy recommend for managers in that department?"
**Query ID**: `f1dde295e3d5`
**Route**: both (follow-up in same Manager session as Example 3a below)
**Refused**: no
**Source**: `evidence/run_live_checks_v2/query_results_v2.jsonl`

**Raw output** (actual, verbatim from evidence):
> The data shows that the Sales department has the lowest average employee satisfaction score at 6.53, making it the department most in need of attention. According to company policy (HR-005), managers in low-scoring departments face direct consequences and required actions: their effectiveness scores from direct reports are factored into their own performance reviews, meaning the low satisfaction scores will impact Sales managers' evaluations directly. According to company policy, if the department's eNPS falls in the 'Watch' range (0 to +19), a CPO-led review is triggered and a targeted action plan must be completed within 30 days; if eNPS falls below 0 ('Critical'), an executive-level response is required. Note that the full policy text was not entirely retrievable — there may be additional recommendations such as coaching programmes or further action-planning requirements that are not captured here, and a full reading of HR-005 is recommended for a complete picture.

**Evidence**:
- Citations: `employee_satisfaction_policy.md::chunk_3` (effectiveness scores in performance reviews), `employee_satisfaction_policy.md::chunk_6` (eNPS Watch/Critical thresholds), all substring-verified
- SQL: `SELECT department, ROUND(AVG(satisfaction_score), 2) AS avg_satisfaction FROM employees WHERE status = 'active' GROUP BY department ORDER BY avg_satisfaction ASC LIMIT 1`
- Result rows: `[{"department": "Sales", "avg_satisfaction": 6.53}]`

**Validation outcome**: valid=True, errors=[], warnings=[]
Tokens: 5,586 in / 1,216 out | Cost: $0.038498

**Human reviewer correction** (confirmed by project author):
> The Sales score of 6.53 is correct and sourced from the database. However, the eNPS passages in the answer are misapplied: the department satisfaction score (1–10 scale) is a different metric from eNPS (−100 to +100), which is computed from a separate survey question not represented in this dataset. The model cannot infer eNPS status from an average satisfaction score. Additionally, the two-quarter coaching trigger requires two consecutive quarters below the 25th percentile — that cannot be determined from a single-period average. The correct summary: Sales has the lowest current satisfaction score (6.53); managers should review recent survey feedback with HR and document actions. Whether the eNPS thresholds or the coaching programme have been triggered cannot be determined from the available data alone.

**Review decision** (confirmed by project author): ACCEPT with reviewer correction applied
**Rationale**: The SQL result and Sales score of 6.53 are verified. The eNPS and coaching-trigger claims are grounded in citation substrings that passed the excerpt check — the validation system correctly confirmed the text appeared in the policy — but the model drew an inference that the policy text does not support in this context. This is an example of the documented limitation: excerpt substring matching confirms text was retrieved, not that every claim derived from it is correct. The reviewer correction provides the accurate interpretation.

---

## Submission Evidence

`evidence/run_live_checks.py` — script that captures 6 live queries through the full application stack.
`evidence/live_calls.jsonl` — 29 Bedrock call records from the original run (pre-fix): raw response text, stop_reason, tokens, cost. Includes the fenced SQL responses that exposed the code-fence parsing bug.
`evidence/query_results.jsonl` — 8 query result records from the original run.

`evidence/run_live_checks_v2/run.py` — post-fix evidence script: combined "both"-route query followed by an ambiguous follow-up in the same Manager session.
`evidence/run_live_checks_v2/live_calls_v2.jsonl` — 12 Bedrock call records from the post-fix run.
`evidence/run_live_checks_v2/query_results_v2.jsonl` — 2 query result records (Q1_both, Q2_followup); both pass validation.

Run `python evidence/run_live_checks.py` or `python evidence/run_live_checks_v2/run.py` to reproduce (requires AWS Bedrock credentials configured via `aws configure` or environment variables).

*All data is entirely fictional. All outputs in this README are from real Bedrock API calls copied verbatim from saved evidence files.*
