"""Structured generation: "give me an instance of this Pydantic model" from an LLM.

Why this file exists
--------------------
The agent must post comments GitHub accepts, so every LLM answer has to be
machine-readable. Asking politely ("please reply in JSON") fails a few percent of
the time: trailing prose, a missing field, a broken quote. This wrapper makes
that failure rate approach zero with three layers:

1. **Constrained decoding via Outlines** (primary). We hand Outlines our Pydantic
   schema; it sends it to the provider as a `json_schema` response format, and
   the provider only lets the model emit tokens that keep the JSON valid. With a
   self-hosted model (vLLM, transformers) Outlines does that token masking itself;
   with Gemini/OpenAI, the provider does it server-side. Our code is the same
   either way. That is the point of using Outlines as the abstraction.
2. **JSON-mode fallback.** Gemini supports only a subset of JSON Schema. If it
   rejects our schema (HTTP 400), we fall back to plain JSON mode with the schema
   written into the prompt, and remember that for later calls.
3. **Validate + one repair.** Every answer is validated by Pydantic (providers
   don't enforce things like `0 <= confidence <= 1`). If it fails, we show the
   model its own answer and the error once and ask for a fix.

Every step is counted in `LLMStats`. That gives you a real number for interviews:
"X% of calls were valid on the first try, Y% needed a repair".
"""

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field, fields
from typing import Any, Protocol, TypeVar

from langsmith import traceable
from outlines.exceptions import APIError, BadRequestError
from outlines.inputs import Chat
from outlines.types import JsonSchema
from pydantic import BaseModel, ValidationError
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from pr_review.config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
Messages = list[dict[str, str]]


class StructuredOutputError(RuntimeError):
    """The model could not produce valid output even after a repair attempt."""


class OutlinesModel(Protocol):
    """What we need from an Outlines model: call it with (input, output_type)."""

    def __call__(self, model_input: Any, output_type: Any = None, **kwargs: Any) -> Any: ...


@dataclass
class LLMStats:
    calls: int = 0  # generate() calls (one per "question" we ask)
    valid_first_try: int = 0  # parsed + validated with no repair needed
    json_mode_calls: int = 0  # calls answered in fallback JSON mode
    repairs: int = 0  # calls that needed the repair round-trip
    failures: int = 0  # calls that failed even after repair
    retries: int = 0  # transient errors (429/5xx/timeout) that we retried
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, **amounts: float) -> None:
        # Graph nodes run in parallel threads, so updates must be locked.
        with self._lock:
            for key, value in amounts.items():
                setattr(self, key, getattr(self, key) + value)

    def as_dict(self) -> dict[str, float]:
        # (not dataclasses.asdict: it deep-copies fields and a Lock can't be copied)
        data = {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}
        data["schema_valid_rate"] = round(self.valid_first_try / self.calls, 4) if self.calls else 0
        return data


def clean_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic schema -> the plain subset every provider accepts.

    - inline `$defs`/`$ref` (Gemini handles references poorly)
    - drop `title`/`default` keywords (noise; some providers reject `default`)
    Careful: a *property* may itself be called "title", so we never drop keys
    that are field names inside `properties`.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node: Any, is_properties: bool = False) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(dict(defs[node["$ref"].rsplit("/", 1)[-1]]))
        out = {}
        for key, value in node.items():
            if not is_properties and key in ("title", "default"):
                continue
            out[key] = walk(value, is_properties=(key == "properties" and not is_properties))
        return out

    return walk(schema)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_json(raw: Any, schema: type[T]) -> T:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise ValueError("empty response (possibly blocked or truncated)")
    text = raw if isinstance(raw, str) else json.dumps(raw)
    text = _FENCE.sub("", text.strip())  # JSON mode sometimes wraps output in ```json
    return schema.model_validate_json(text)


def _is_retryable(error: BaseException) -> bool:
    return isinstance(error, APIError) and error.retryable


class StructuredLLM:
    def __init__(
        self,
        settings: Settings,
        model: OutlinesModel | None = None,
        http_client: Any = None,  # tests inject a fake HTTP transport here
    ) -> None:
        self.settings = settings
        self.stats = LLMStats()
        self.schema_supported = True
        self._model = model  # built lazily: cached reviews never need an API key
        self._http_client = http_client

    @property
    def model(self) -> OutlinesModel:
        if self._model is None:
            self._model = self._build_model(self.settings)
        return self._model

    def _build_model(self, settings: Settings) -> OutlinesModel:
        import openai
        import outlines
        from langsmith.wrappers import wrap_openai

        if settings.llm_api_key is None:
            raise RuntimeError("LLM_API_KEY is not set. Run `pr-review doctor` for help.")
        client = openai.OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            timeout=settings.llm_timeout_s,
            max_retries=0,  # we retry ourselves (below) so backoff is visible and counted
            http_client=self._http_client,
        )
        # wrap_openai: every HTTP request shows up in LangSmith with tokens + latency.
        client = wrap_openai(client)
        self._count_usage(client)
        return outlines.from_openai(client, settings.llm_model)

    def _count_usage(self, client: Any) -> None:
        create = client.chat.completions.create

        def counting_create(*args: Any, **kwargs: Any) -> Any:
            result = create(*args, **kwargs)
            usage = getattr(result, "usage", None)
            if usage is not None:
                self.stats.add(
                    prompt_tokens=usage.prompt_tokens or 0,
                    completion_tokens=usage.completion_tokens or 0,
                )
            return result

        client.chat.completions.create = counting_create

    # ----------------------------------------------------------------- public
    @traceable(run_type="chain", name="structured_generate")
    def generate(self, messages: Messages, schema: type[T], *, name: str = "") -> T:
        """Ask the model and return a validated `schema` instance."""
        self.stats.add(calls=1)
        started = time.perf_counter()
        try:
            return self._generate(messages, schema)
        finally:
            self.stats.add(latency_s=time.perf_counter() - started)

    # --------------------------------------------------------------- internals
    def _generate(self, messages: Messages, schema: type[T]) -> T:
        json_mode = not self.schema_supported
        if not json_mode:
            try:
                raw = self._call(messages, JsonSchema(clean_schema(schema)))
            except BadRequestError as error:
                log.warning("Provider rejected the JSON schema; using JSON mode. (%s)", error)
                json_mode = True
        if json_mode:
            messages = self._with_schema_in_prompt(messages, schema)
            raw = self._call(messages, dict)
            self.stats.add(json_mode_calls=1)
            # A JSON-mode call worked where the schema call failed -> the schema was
            # the problem. Skip the doomed schema attempt from now on.
            self.schema_supported = False

        try:
            result = parse_json(raw, schema)
            self.stats.add(valid_first_try=1)
            return result
        except (ValidationError, ValueError) as error:
            return self._repair(messages, raw, error, schema, json_mode)

    def _repair(
        self, messages: Messages, raw: Any, error: Exception, schema: type[T], json_mode: bool
    ) -> T:
        self.stats.add(repairs=1)
        repair = [
            *messages,
            {"role": "assistant", "content": str(raw)},
            {
                "role": "user",
                "content": f"That output was invalid:\n{error}\n"
                "Return the corrected JSON only, matching the schema exactly.",
            },
        ]
        output_type = dict if json_mode else JsonSchema(clean_schema(schema))
        try:
            return parse_json(self._call(repair, output_type), schema)
        except (ValidationError, ValueError) as second_error:
            self.stats.add(failures=1)
            raise StructuredOutputError(
                f"{schema.__name__}: invalid output after repair: {second_error}"
            ) from second_error

    def _call(self, messages: Messages, output_type: Any) -> Any:
        for attempt in Retrying(
            retry=retry_if_exception(_is_retryable),
            wait=wait_random_exponential(multiplier=2, max=60),
            stop=stop_after_attempt(self.settings.llm_max_attempts),
            before_sleep=lambda state: self.stats.add(retries=1),
            reraise=True,
        ):
            with attempt:
                return self.model(
                    Chat(messages), output_type, temperature=self.settings.llm_temperature
                )
        raise AssertionError("unreachable")  # Retrying either returns or re-raises

    @staticmethod
    def _with_schema_in_prompt(messages: Messages, schema: type[T]) -> Messages:
        instruction = (
            "Respond with a single JSON object (no prose, no code fences) that matches "
            f"this JSON Schema exactly:\n{json.dumps(clean_schema(schema))}"
        )
        # Merge into an existing system message: some providers dislike two of them.
        if messages and messages[0]["role"] == "system":
            first = {"role": "system", "content": f"{messages[0]['content']}\n\n{instruction}"}
            return [first, *messages[1:]]
        return [{"role": "system", "content": instruction}, *messages]
