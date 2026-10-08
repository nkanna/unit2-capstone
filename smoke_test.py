"""One Gemini smoke query; no retrieval, SQL, or full live evidence suite.
Credentials and provider error bodies are never printed or saved.
"""
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from llm_client import LLMAPIError, invoke
from tokenomics.logger import log_call, log_query_summary, new_query_id

EVIDENCE_PATH = Path(__file__).resolve().parent / 'evidence/gemini/smoke_test.jsonl'

def main() -> int:
    query = 'Say hello in exactly one sentence.'
    record = {'timestamp': datetime.now(timezone.utc).isoformat(), 'provider': 'google_gemini'}
    code = 1
    try:
        response = invoke(messages=[{'role': 'user', 'content': query}], max_tokens=256)
        usage_complete = all(v is not None for v in [response.input_tokens, response.output_tokens, response.total_tokens])
        passed = response.stop_reason == 'end_turn' and not response.response_issue and usage_complete
        qid = new_query_id()
        call = log_call(qid, 'smoke_test', response, 'pass' if passed else 'fail')
        log_query_summary(qid, query, [call])
        record.update(status='pass' if passed else 'fail', query_id=qid, response=asdict(response))
        if not usage_complete:
            record['error'] = 'Provider response did not include complete usage metadata.'
        code = 0 if passed else 1
    except LLMAPIError as exc:
        record.update(status='fail', error=str(exc), status_code=exc.status_code)
    except Exception as exc:
        record.update(status='fail', error=f'Smoke test could not complete ({type(exc).__name__}).')
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVIDENCE_PATH.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return code

if __name__ == '__main__':
    raise SystemExit(main())
