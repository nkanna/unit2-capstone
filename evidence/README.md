# Evidence provenance

Historical, read-only Bedrock/Claude evidence:
- `live_calls.jsonl` and `query_results.jsonl`
- `run_live_checks_v2/live_calls_v2.jsonl` and `query_results_v2.jsonl`
- Project-root `tokenomics_log.jsonl`

These files are unchanged and support the historical README human-review
examples. They do not demonstrate Gemini results. Historical human-review decisions
confirmed by the project author are recorded in the README.

New Gemini evidence lives in `gemini/`; the smoke test writes
`gemini/smoke_test.jsonl`. The full evidence runners now write separate files
there. The author ran an initial six-query batch with service/JSON failures;
the current verified runs are listed below. Real usage is logged only
when returned by the provider. Missing credentials are recorded honestly.

## Corrected Gemini capture

`python evidence/run_live_checks.py` runs ten unique queries. Every invocation
creates `evidence/gemini/runs/<run-id>/` with `live_calls.jsonl`,
`query_results.jsonl`, and `tokenomics.jsonl`. Earlier Gemini and Bedrock files
are preserved. Partial responses and sanitized API failures are retained;
failed API attempts have unknown usage, never fabricated token counts.

## Verified live runs (2026-10-08)

- `gemini/runs/20261008T174810Z-cb8972bf/`: 9/10 checks passed; Q4 failed with 503.
- `gemini/runs/20261008T180319Z-0146155d/`: 2/2 passed, combined and follow-up in one Manager session.
- All ten distinct user questions have successful results across the two runs.
- Uploaded source JSONL files are unchanged. Failed batches, smoke, and historical
  Bedrock logs remain separate. Current human-review examples are in the README.
