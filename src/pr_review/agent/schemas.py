"""Shapes the LLM must answer in (enforced by Outlines).

Design choices that raise precision, and that you can explain in an interview:

- `category` is a closed list with **no "style" or "nit" option**. The model
  cannot even express a style complaint, so whole classes of low-value comments
  (naming, formatting, "consider adding a docstring") disappear by construction.
- `quote` forces the model to copy the exact line it is complaining about. We
  check that quote against the real diff (grounding), which catches invented issues.
- `analysis` / `reasoning` come *before* the answer fields. The model writes
  JSON top to bottom, so it reasons first and decides second. Constraining the
  output format can hurt reasoning quality if there's no room to think.
- Every field is required (no defaults). Strict structured-output modes require
  that; "no suggestion" is expressed as an empty string instead of null.
"""

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["bug", "security", "error-handling", "missing-test", "performance"]
Severity = Literal["low", "medium", "high", "critical"]


class Issue(BaseModel):
    path: str = Field(description="File path exactly as shown after '###' in the diff.")
    line: int = Field(description="Line number shown at the left of the '+' line.")
    quote: str = Field(description="The exact code copied from that '+' line.")
    category: Category
    severity: Severity
    title: str = Field(description="One short sentence naming the problem.")
    explanation: str = Field(
        description="Why it is wrong and what concretely breaks (inputs -> bad outcome)."
    )
    suggested_code: str = Field(
        description="Replacement for that single line, or empty string if none."
    )
    confidence: float = Field(
        ge=0, le=1, description="Probability a senior engineer would agree this is a real issue."
    )


class IssueList(BaseModel):
    analysis: str = Field(description="Brief notes: what the change does and where risk is.")
    issues: list[Issue]


class Verdict(BaseModel):
    issue_id: int
    reasoning: str = Field(description="Try to disprove the issue first, then conclude.")
    verdict: Literal["keep", "drop"]
    confidence: float = Field(ge=0, le=1)


class VerifyResult(BaseModel):
    verdicts: list[Verdict]


class Candidate(BaseModel):
    """An Issue plus bookkeeping the agent adds as it moves through the graph."""

    id: int
    issue: Issue
    chunk: int
    original_confidence: float
    # "" = not verified yet, "unverified" = verifier call failed (kept, penalised)
    verdict: Literal["", "keep", "drop", "unverified"] = ""
    verifier_reasoning: str = ""
