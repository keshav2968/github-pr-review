"""Integration test of the REAL stack (Outlines -> wrap_openai -> openai SDK -> HTTP)
against a fake OpenAI-compatible server. Catches API misuse without a key or network."""

import json

import httpx

from pr_review.agent.schemas import IssueList
from pr_review.config import Settings
from pr_review.llm.structured import StructuredLLM


def completion(content: str) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "gemini-test",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    }


def make_llm(handler):
    settings = Settings(
        _env_file=None, llm_api_key="test-key", llm_model="gemini-test",
        llm_base_url="https://fake.local/v1/", llm_max_attempts=2,
    )  # fmt: skip
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return StructuredLLM(settings, http_client=http)


def test_request_shape_and_token_counting():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json=completion('{"analysis": "ok", "issues": []}'))

    llm = make_llm(handler)
    out = llm.generate([{"role": "user", "content": "review"}], IssueList)

    assert out.issues == []
    body = seen[0]
    assert body["model"] == "gemini-test"
    assert body["temperature"] == 0.2
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert "$defs" not in json.dumps(fmt)  # our cleaned schema was sent
    assert llm.stats.prompt_tokens == 120 and llm.stats.completion_tokens == 30


def test_schema_rejection_then_json_mode_over_http():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, json={"error": {"message": "Invalid JSON schema"}})
        return httpx.Response(200, json=completion('{"analysis": "ok", "issues": []}'))

    llm = make_llm(handler)
    llm.generate([{"role": "user", "content": "review"}], IssueList)
    assert not llm.schema_supported and llm.stats.json_mode_calls == 1
