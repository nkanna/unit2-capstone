# Final Gemini Verification — 2026-10-08

The migration is verified against the project author's uploaded live evidence.
- **236 offline tests passed on the author's Mac in 1.97s**. The container also
  passed 236 tests before this documentation-only finalization.
- Ten-query run: **9/10 passed** (`20261008T174810Z-cb8972bf`).
- Same-session combined/follow-up retry: **2/2 passed**
  (`20261008T180319Z-0146155d`). All **ten distinct** questions now have successful
  results; no single 10/10 run is claimed.
- Six captured SQL result sets match the committed SQLite database.
- Sixteen citation IDs/excerpts match captured chunks and complete source policy
  documents, with whitespace normalization. Chunk-boundary fragments are a
  documented readability limitation, not proof of policy applicability.
- Follow-up `history_before` exactly equals the combined answer's `history_after`
  and contains that prior answer, in one Manager instance.
- All 33 returned calls in the two verified runs have finish reason `stop`,
  preserved usage, and passing automated validation. Per-query summaries
  reconcile to the call ledgers. The failed follow-up in the first run has no
  returned response and therefore no query usage summary; its 503 attempts
  remain in raw capture with unknown usage.
- Verified returned-response totals: 28,030 input + 4,952 answer + 0 derived
  thinking = 32,982 total tokens. Hypothetical paid-tier cost: **$0.02078900**
  using Flash-Lite rates. Billing tier is unknown; no actual charge is claimed.
- The earlier Gemini SQL failure was confirmed as `length`/`max_tokens` from
  saved raw evidence. It is distinct from the historical Bedrock fence bug.
- Three current Gemini Trust-but-Verify examples are documented. The author's
  previously confirmed review standard is retained: accept PR rules and revenue
  after verification; clarify that Sales' ranking does not establish policy
  coaching/eNPS thresholds. Raw output and human clarification are separate.
- `gemini-3.6-flash` remains the specified default; `gemini-3.5-flash-lite` is the
  instructor-approved model used for the full live evidence. Set it in the
  local environment to reproduce that model choice.
- No further code changes, live calls, commits or pushes were performed during
  final verification. All uploaded live evidence is included unmodified.

The chronological implementation reports below describe earlier milestones and
outstanding checks at those times. This final verification supersedes their
pending-live statements. See README's verified-results section for submission.

---

# Gemini Migration Report

Migrated the uploaded `unit2-capstone-final-review.zip` implementation in place
without rebuilding the application. Google Gemini is the model provider; the
OpenAI Python SDK is only the client for Google's compatibility endpoint.

## Actual results

| Check | Result |
|---|---|
| Complete offline suite: `python3 -m pytest tests/ -q` | **236 passed in 1.68s** |
| Existing regression tests | All 165 retained and passed |
| New tests | 71 migration tests, including real qualitative-agent/logger integration |
| Python syntax: `python3 -m compileall -q project` from parent folder | Passed |
| Runtime used for checks | Python 3.12.14 and OpenAI SDK 2.54.0 |
| Real smoke command: `python3 smoke_test.py` | **Failed, exit 1: GEMINI_API_KEY unavailable** |
| Live Gemini API request | **Not sent**; credential check failed first |
| New real Gemini response/token usage | None; no token totals fabricated |
| Full live evidence suite | Not run |
| Commit / push | Not performed |

The saved smoke failure is in `evidence/gemini/smoke_test.jsonl`. Mock responses
are used only in offline tests and are never represented as live evidence.
No `.env` file or real credential is included in this deliverable.

## Second live pair and timeout correction

The author supplied output from the legacy two-query runner: the combined
question timed out at the 30-second request limit; the follow-up then exhausted
three HTTP 503 attempts. This pair did not establish successful session history.

- Extend the per-attempt SDK timeout to 90 seconds and retry SDK timeout errors
  within the same three-request limit. Timeout attempts retain unknown usage.
- Add `--followup-pair` to the current runner. It uses one Manager instance,
  writes a unique new evidence directory, and skips the follow-up if the
  prerequisite combined question fails. It does not overwrite previous runs.
- Add timeout-success, timeout-exhaustion, and failed-context-skip regressions.
  Current complete offline suite: **236 passed in 1.68s**.
- The author's earlier ten-query run passed nine checks, including policies,
  revenue, combined synthesis, refusals, active employee counts and churn. Its
  follow-up failed with HTTP 503. Those are reported terminal results, awaiting
  inspection of the raw evidence ZIP; no passing follow-up is claimed.

## First full Gemini batch and reliability correction

The author ran the initial six-query capture on their Mac after **226 offline
tests passed**. Ingestion completed with 39 chunks. Q1, Q3 and Q4 failed with
HTTP 503; Q2's SQL generation failed with an unterminated JSON string; Q5 and Q6
produced valid refusals. The successful simple smoke did not establish that the
full agent pipeline worked. No passing full Gemini batch is claimed.

- Retry HTTP 503 with bounded exponential backoff, up to three total requests.
  Retry the selected model for service errors; only two quota errors trigger
  automatic fallback. Preserve sanitized failed-attempt metadata.
- Increase classification/SQL generation budgets to 4096 and policy,
  interpretation and synthesis budgets to 8192. Gemini's cap includes thinking
  and visible output. Truncation is a plausible cause of the SQL failure, not a
  proven diagnosis without checking the saved raw response and finish reason.
- Replace the old six-query capture with ten unique queries, keeping the same
  combined-plus-follow-up sequence. Each run has a new directory and its own
  usage ledger. Checkpoint after every query, keep real partial calls and
  failed-request metadata, and return failure when in-scope answers fail.
- Tests now cover service-error recovery/exhaustion, mixed quota/service errors,
  failed-query partial accounting, non-overwritten runs, unexpected refusals,
  and the ten-query/follow-up sequence. **233 offline tests passed**.
- New file: `tests/test_live_capture.py`. No additional live API request was
  made in the container. The corrected runner needs a live run on the Mac.

## Project author's live verification (Mac)

The author supplied terminal output showing **222 tests passed in 2.90s** and a
successful live smoke request at `2026-10-08T17:32:57.517772+00:00`, query ID
`36d00d4246a9`, model `gemini-3.6-flash`, finish reason `stop`, and no response issue.
The returned provider usage was 8 prompt tokens, 10 answer tokens, and 257 total
tokens. No token values were changed to make the totals add up.

The container's original missing-key smoke failure remains preserved in the
archive. The successful smoke evidence is on the author's Mac; this report
records the supplied output, not an additional live request made here. Full
end-to-end Gemini agent evidence remains outstanding.

## Corrections after successful smoke

- Gemini's compatibility endpoint excludes thinking from completion tokens but
  includes it in total tokens. The original paid estimate used only answer
  tokens, so it underestimated hypothetical paid usage.
- For this text-only client with no provider tools, derive thinking as
  `total - prompt - completion`, label it `derived_from_provider_totals`, and
  price all generated tokens (`total - prompt`) once. The author's call thus
  has 239 derived thinking tokens and 249 billable output tokens. Returned
  provider usage remains unchanged; these derived fields do not pretend to be
  explicitly reported thoughts-token metadata.
- Missing or inconsistent totals retain unknown accounting and paid estimates.
- Added four accounting regressions: the observed live usage shape, reasoning
  detail not counted twice, missing total, and inconsistent total. Current
  container suite: **236 passed in 1.68s**.

## Corrections after ZIP review

- Fixed the qualitative agent's logger call from the obsolete `bedrock_response`
  keyword to `response`. Previously a real policy query raised `TypeError`;
  mocks in the original tests hid the interface mismatch.
- Added two offline integration checks using the real qualitative agent and
  real logger with mocked retrieval/provider responses: successful generation,
  and truncation followed by a valid retry. Both validate persisted records.
- Removed `temperature` from the Google request while retaining the shared
  argument for existing callers. The fake HTTP transport check asserts that
  deprecated sampling parameters are absent from the serialized request.
  Reference: https://ai.google.dev/gemini-api/docs/generate-content/whats-new-gemini-3.6.
- Restored the author's previously confirmed historical review decisions in the
  README, corrected the regional-revenue evidence path, and included the raw
  satisfaction follow-up alongside the separate human correction. The raw
  JSONL evidence remains unchanged.
- No live API request, full evidence capture, commit, or push was performed
  during the initial ZIP correction. The author subsequently supplied a
  successful live smoke; the corrected full capture is still pending.

## Implementation

- Every application LLM call uses `llm_client.invoke()` and Google's endpoint
  `https://generativelanguage.googleapis.com/v1beta/openai/`.
- dotenv reads `GEMINI_API_KEY` and `GEMINI_MODEL`; default model is
  `gemini-3.6-flash`. Three bounded attempts retry HTTP 429 and 503. Two quota failures select
  the approved `gemini-3.5-flash-lite` fallback; service errors retain the
  selected model. SDK retries are off; backoff waits are 1 and 2 seconds.
  Other errors fail immediately.
- All five system prompts are unchanged, including SQL schema context.
  User/assistant history, retrieval, SQLite, SQL safety, numeric and citation
  validation, and JSON parsing are retained. The validator and JSON parser's
  executable ASTs match the original; only provider descriptions changed.
- Responses retain provider model, original finish reason, returned usage and
  total tokens. Normal completion maps to `end_turn`; truncation maps to
  `max_tokens`. Missing content, safety blocks and incomplete reasons cannot
  pass as successful responses. Missing token counts remain null.
- Tokenomics identifies the returned model, logs real provider usage and
  per-query totals, and distinguishes conditional free-tier estimates from
  hypothetical paid-tier estimates. Unknown billing tier stays unknown.
  Google Standard pricing was verified on 2026-10-08:
  https://ai.google.dev/gemini-api/docs/pricing
- Gemini Flash current promotional input/output rates are $0.75/$3.75 per
  million tokens through 2026-12-31, then $1.50/$7.50 from 2027-01-01.
  Flash-Lite uses $0.30/$2.50. Cached-input rates are handled separately.
  Claude rates and legacy environment overrides are not reused.
- New tokenomics defaults to `tokenomics_gemini_log.jsonl`. This file is not
  created unless a provider response is actually returned. Smoke failures
  are recorded separately without fake usage. Full evidence scripts now
  write exclusively under `evidence/gemini/`.
- All 12 original data/log files remain byte-for-byte unchanged. The historical
  human-review examples remain verbatim and are labeled historical. They do
  not establish Gemini performance or completion of the live-query requirement.

## Changed files

- `.env.example`
- `README.md`
- `agents/manager.py`
- `agents/qualitative.py`
- `agents/quantitative.py`
- `bedrock_client.py`
- `capstone_requirements.md`
- `check_retrieval.py`
- `evidence/run_live_checks.py`
- `evidence/run_live_checks_v2/run.py`
- `main.py`
- `requirements.txt`
- `smoke_test.py`
- `tests/test_manager.py`
- `tests/test_qualitative.py`
- `tests/test_quantitative.py`
- `tests/test_validator.py`
- `tokenomics/logger.py`
- `utils.py`
- `validation/validator.py`

## Added files

- `evidence/README.md`
- `evidence/gemini/smoke_test.jsonl`
- `llm_client.py`
- `tests/test_llm_client.py`
- `tests/test_smoke_test.py`
- `tests/test_tokenomics.py`
- `MIGRATION_REPORT.md` (this report)

## Remaining live verification

Apply the changed/new source files to your existing project, preserving your
local `.env`, Chroma storage and data. Replace the example placeholders in
`.env` locally, set `GEMINI_MODEL=gemini-3.6-flash` (or leave it blank), and set
the billing tier only if known. Do not share or display the key.

From that project folder:

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python smoke_test.py
```

The smoke script alone will make one query, with bounded quota retries if
needed. It requires nonempty completed content and real usage to pass.
The author has supplied a successful authenticated live smoke result. After
applying this accounting correction, rerun the offline tests and optionally
repeat the smoke to produce a record with the corrected accounting. Then run `GEMINI_MODEL=gemini-3.5-flash-lite python evidence/run_live_checks.py --followup-pair`
for a new two-query capture alongside the author's preserved nine passing checks.
Review the saved results before committing or pushing.
