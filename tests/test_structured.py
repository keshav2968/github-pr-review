import pytest
from outlines.exceptions import BadRequestError, RateLimitError
from outlines.types import JsonSchema

from pr_review.agent.schemas import IssueList
from pr_review.config import Settings
from pr_review.llm.structured import StructuredLLM, StructuredOutputError, clean_schema

GOOD = '{"analysis": "fine", "issues": []}'


class ScriptedModel:
    """Fake Outlines model: returns/raises the scripted items in order."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def __call__(self, model_input, output_type=None, **kwargs):
        self.calls.append(output_type)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def llm(*script):
    model = ScriptedModel(*script)
    settings = Settings(_env_file=None, llm_max_attempts=3)
    return StructuredLLM(settings, model=model), model


def test_clean_schema_inlines_refs_and_keeps_field_named_title():
    schema = clean_schema(IssueList)
    assert "$defs" not in schema and "$ref" not in str(schema)
    item = schema["properties"]["issues"]["items"]
    assert "title" in item["properties"]  # the FIELD 'title' survives...
    assert "title" not in item  # ...while the 'title' keyword is removed
    assert item["properties"]["category"]["enum"][0] == "bug"


def test_schema_mode_success():
    client, model = llm(GOOD)
    assert client.generate([{"role": "user", "content": "x"}], IssueList).issues == []
    assert isinstance(model.calls[0], JsonSchema)
    assert client.stats.as_dict()["schema_valid_rate"] == 1.0


def test_code_fences_are_stripped():
    client, _ = llm(f"```json\n{GOOD}\n```")
    assert client.generate([{"role": "user", "content": "x"}], IssueList).analysis == "fine"


def test_rate_limit_is_retried(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)
    client, model = llm(RateLimitError("429"), GOOD)
    client.generate([{"role": "user", "content": "x"}], IssueList)
    assert client.stats.retries == 1 and len(model.calls) == 2


def test_schema_rejected_falls_back_to_json_mode_and_remembers():
    client, model = llm(BadRequestError("bad schema"), GOOD, GOOD)
    client.generate(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "x"}], IssueList
    )
    assert model.calls[1] is dict and not client.schema_supported
    client.generate([{"role": "user", "content": "x"}], IssueList)
    assert model.calls[2] is dict  # no doomed schema attempt the second time
    assert client.stats.json_mode_calls == 2


def test_invalid_output_is_repaired_once():
    bad = '{"analysis": "x", "issues": [{"path": "a"}]}'
    client, _ = llm(bad, GOOD)
    client.generate([{"role": "user", "content": "x"}], IssueList)
    assert client.stats.repairs == 1 and client.stats.valid_first_try == 0


def test_gives_up_after_failed_repair():
    client, _ = llm("not json", "still not json")
    with pytest.raises(StructuredOutputError):
        client.generate([{"role": "user", "content": "x"}], IssueList)
    assert client.stats.failures == 1
