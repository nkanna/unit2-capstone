"""Client contract tests use only fake transports and dummy credentials."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock
import httpx
import pytest
from openai import OpenAI, APITimeoutError
import llm_client as lc


def completion(reason='stop', content='{"answer":"ok"}', usage=None, model=lc.DEFAULT_MODEL):
    return {'model':model, 'choices':[{'finish_reason':reason,'message':{'content':content}}],
            'usage': usage if usage is not None else {'prompt_tokens':11,'completion_tokens':7,'total_tokens':18}}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('GEMINI_MODEL', raising=False)
    monkeypatch.setattr(lc, '_client', None)
    for name in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
        monkeypatch.delenv(name, raising=False)


def fake_client(monkeypatch, values):
    client = MagicMock()
    client.chat.completions.create.side_effect = values
    monkeypatch.setattr(lc, '_get_client', lambda:client)
    return client


def test_sdk_transport_targets_google_and_preserves_system_and_history(monkeypatch):
    captured=[]
    def handler(request):
        captured.append(request)
        return httpx.Response(200, json={'id':'test', 'object':'chat.completion','created':0, **completion()})
    client=OpenAI(api_key='offline-dummy-key', base_url=lc.BASE_URL, max_retries=0,
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(lc,'_get_client',lambda:client)
    messages=[{'role':'user','content':'earlier Q'}, {'role':'assistant','content':'earlier A'}, {'role':'user','content':'follow-up'}]
    original=[dict(x) for x in messages]
    result=lc.invoke(messages, system='Keep this system instruction.', max_tokens=512, temperature=0.0)
    request=captured[0]
    assert str(request.url)==lc.BASE_URL+'chat/completions'
    body=json.loads(request.content)
    assert body['messages']==[{'role':'system','content':'Keep this system instruction.'}]+original
    assert body['model']==lc.DEFAULT_MODEL
    assert body['max_tokens']==512
    assert 'temperature' not in body
    assert 'top_p' not in body and 'top_k' not in body
    assert messages==original
    assert (result.input_tokens,result.output_tokens,result.total_tokens)==(11,7,18)
    assert result.provider_finish_reason=='stop' and result.stop_reason=='end_turn'
    client.close()


def test_environment_model_override(monkeypatch):
    monkeypatch.setenv('GEMINI_MODEL',lc.QUOTA_FALLBACK_MODEL)
    client=fake_client(monkeypatch,[completion(model=lc.QUOTA_FALLBACK_MODEL)])
    assert lc.invoke([]).model==lc.QUOTA_FALLBACK_MODEL
    assert client.chat.completions.create.call_args.kwargs['model']==lc.QUOTA_FALLBACK_MODEL


@pytest.mark.parametrize('reason,normalized,issue',[
    ('stop','end_turn',None), ('STOP','end_turn',None),
    ('length','max_tokens','truncated_response'), ('MAX_TOKENS','max_tokens','truncated_response'),
    ('content_filter','safety_block','safety_block'),('SAFETY','safety_block','safety_block'),
    ('RECITATION','safety_block','safety_block'),('PROHIBITED_CONTENT','safety_block','safety_block'),
    ('tool_calls','tool_calls','incomplete_response'), (None,'missing_finish_reason','incomplete_response'),
    ('unknown_value','unknown_value','incomplete_response'),
])
def test_finish_reasons_preserve_provider_value(reason,normalized,issue):
    r=lc._map_response(completion(reason),lc.DEFAULT_MODEL)
    assert r.stop_reason==normalized
    assert r.provider_finish_reason==reason
    assert r.response_issue==issue
    assert r.input_tokens==11
    if issue=='safety_block': assert r.text==''
    if issue=='truncated_response': assert r.text=='{"answer":"ok"}'


@pytest.mark.parametrize('content',[None,'','   ',[],[{'type':'text','text':'unsupported block'}]])
def test_missing_content_is_not_a_success(content):
    r=lc._map_response(completion(content=content),lc.DEFAULT_MODEL)
    assert r.stop_reason=='missing_content' and r.response_issue=='missing_content'
    assert r.provider_finish_reason=='stop'


def test_missing_choices_and_prompt_safety_block():
    r=lc._map_response({'usage':{'prompt_tokens':9},'choices':[], 'promptFeedback':{'blockReason':'SAFETY'}},lc.DEFAULT_MODEL)
    assert r.response_issue=='safety_block'
    assert r.input_tokens==9 and r.output_tokens is None and r.total_tokens is None


def test_message_refusal_is_blocked():
    p=completion(); p['choices'][0]['message']['refusal']='blocked'
    assert lc._map_response(p,lc.DEFAULT_MODEL).stop_reason=='safety_block'


def test_missing_usage_and_model_are_not_invented():
    r=lc._map_response({'choices':[{'finish_reason':'stop','message':{'content':'ok'}}]},lc.QUOTA_FALLBACK_MODEL)
    assert r.input_tokens is None and r.output_tokens is None and r.total_tokens is None
    assert r.usage_metadata=={} and not r.model_reported
    assert r.model==lc.QUOTA_FALLBACK_MODEL


def test_usage_metadata_retained_without_adding_reasoning_twice():
    usage={'prompt_tokens':11,'completion_tokens':7,'total_tokens':18,
           'completion_tokens_details':{'reasoning_tokens':4},'prompt_tokens_details':{'cached_tokens':3}}
    r=lc._map_response(completion(usage=usage),lc.DEFAULT_MODEL)
    assert r.usage_metadata==usage and r.output_tokens==7 and r.total_tokens==18


def test_camel_case_usage_mapping():
    r=lc._map_response(completion(usage={'promptTokens':0,'completionTokens':2,'totalTokens':2}),lc.DEFAULT_MODEL)
    assert (r.input_tokens,r.output_tokens,r.total_tokens)==(0,2,2)


@pytest.mark.parametrize('invalid',[-1,True,2.5,'12'])
def test_invalid_usage_stays_unknown(invalid):
    r=lc._map_response(completion(usage={'prompt_tokens':invalid}),lc.DEFAULT_MODEL)
    assert r.input_tokens is None


class FakeAPIError(Exception):
    def __init__(self,status):
        super().__init__('SECRET_PROVIDER_BODY offline-dummy-key')
        self.status_code=status


def test_quota_fallback_is_bounded_and_reports_actual_model(monkeypatch):
    sleep=MagicMock(); monkeypatch.setattr(lc.time,'sleep',sleep)
    client=fake_client(monkeypatch,[FakeAPIError(429),FakeAPIError(429),completion(model=lc.QUOTA_FALLBACK_MODEL)])
    r=lc.invoke([])
    assert [c.kwargs['model'] for c in client.chat.completions.create.call_args_list]==[lc.DEFAULT_MODEL,lc.DEFAULT_MODEL,lc.QUOTA_FALLBACK_MODEL]
    assert r.model==lc.QUOTA_FALLBACK_MODEL and len(r.attempts)==3
    assert [c.args[0] for c in sleep.call_args_list]==[1,2]


def test_quota_exhaustion_returns_sanitized_failure(monkeypatch):
    monkeypatch.setattr(lc.time,'sleep',lambda _:None)
    client=fake_client(monkeypatch,[FakeAPIError(429)]*3)
    with pytest.raises(lc.LLMAPIError) as e: lc.invoke([])
    assert client.chat.completions.create.call_count==3
    assert e.value.status_code==429
    assert 'SECRET_PROVIDER_BODY' not in str(e.value) and 'offline-dummy-key' not in str(e.value)


@pytest.mark.parametrize('status',[400,401,403,404,500])
def test_non_quota_api_errors_do_not_retry(monkeypatch,status):
    client=fake_client(monkeypatch,[FakeAPIError(status)])
    with pytest.raises(lc.LLMAPIError): lc.invoke([])
    assert client.chat.completions.create.call_count==1


def test_connection_error_is_sanitized(monkeypatch):
    fake_client(monkeypatch,[RuntimeError('SECRET_PROVIDER_BODY')])
    with pytest.raises(lc.LLMAPIError,match='RuntimeError') as e: lc.invoke([])
    assert 'SECRET_PROVIDER_BODY' not in str(e.value)


def test_missing_key_fails_before_sdk_request():
    with pytest.raises(lc.LLMAPIError,match='missing'): lc.invoke([])


def test_constructed_client_uses_google_and_disables_sdk_retries(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','offline-dummy-key')
    client=lc._get_client()
    assert str(client.base_url)==lc.BASE_URL and client.max_retries==0
    assert client.timeout==90.0
    client.close()

def test_service_unavailable_retries_same_model_then_succeeds(monkeypatch):
    sleep = MagicMock(); monkeypatch.setattr(lc.time, 'sleep', sleep)
    client = fake_client(monkeypatch, [FakeAPIError(503), FakeAPIError(503), completion()])
    result = lc.invoke([])
    assert result.stop_reason == 'end_turn'
    assert [c.kwargs['model'] for c in client.chat.completions.create.call_args_list] == [lc.DEFAULT_MODEL] * 3
    assert [c.args[0] for c in sleep.call_args_list] == [1, 2]

def test_service_unavailable_exhaustion_preserves_safe_attempts(monkeypatch):
    monkeypatch.setattr(lc.time, 'sleep', lambda _: None)
    client = fake_client(monkeypatch, [FakeAPIError(503)] * 3)
    with pytest.raises(lc.LLMAPIError) as error:
        lc.invoke([])
    assert client.chat.completions.create.call_count == 3
    assert [a['status_code'] for a in error.value.attempts] == [503] * 3
    assert 'SECRET_PROVIDER_BODY' not in str(error.value.attempts)

def test_mixed_service_and_quota_failures_do_not_switch_without_two_quotas(monkeypatch):
    monkeypatch.setattr(lc.time, 'sleep', lambda _: None)
    client = fake_client(monkeypatch, [FakeAPIError(503), FakeAPIError(429), completion()])
    lc.invoke([])
    assert [c.kwargs['model'] for c in client.chat.completions.create.call_args_list] == [lc.DEFAULT_MODEL] * 3

def test_timeout_retries_same_model_and_succeeds(monkeypatch):
    monkeypatch.setattr(lc.time, 'sleep', lambda _: None)
    timeout = APITimeoutError(request=httpx.Request('POST', lc.BASE_URL))
    client = fake_client(monkeypatch, [timeout, completion()])
    result = lc.invoke([])
    assert client.chat.completions.create.call_count == 2
    assert result.attempts[0]['error_type'] == 'timeout'
    assert all(a['model'] == lc.DEFAULT_MODEL for a in result.attempts)

def test_timeout_exhaustion_remains_bounded_and_preserves_unknown_usage(monkeypatch):
    monkeypatch.setattr(lc.time, 'sleep', lambda _: None)
    timeout = APITimeoutError(request=httpx.Request('POST', lc.BASE_URL))
    client = fake_client(monkeypatch, [timeout] * 3)
    with pytest.raises(lc.LLMAPIError) as error:
        lc.invoke([])
    assert client.chat.completions.create.call_count == 3
    assert all(a['error_type'] == 'timeout' for a in error.value.attempts)
    assert not any('tokens' in key for attempt in error.value.attempts for key in attempt)
