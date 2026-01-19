from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ReviewError(BaseModel):
    location: str              # "Step 5", "Assumption about Mike", etc.
    error_type: Literal[
        "logical_error",
        "missing_case",
        "invalid_assumption",
        "math_error",
        "inconsistency",
        "unclear_reasoning"
    ]
    description: str
    severity: Literal["minor", "major", "critical"]

class PeerEvaluation(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    errors: list[ReviewError]
    suggested_changes: list[str]

class ProblemSolutionReview(BaseModel):
    review_id: str | None = Field(
        default=None,
        exclude=True
    )
    run_id: str | None = Field(
        default=None,
        exclude=True
    )
    problem_id: str| None = Field(
        default=None,
        exclude=True
    )

    reviewer_id: str | None = Field(
        default=None,
        exclude=True
    )
    reviewee_id: str | None = Field(
        default=None,
        exclude=True
    )

    evaluation: PeerEvaluation

    overall_assessment: Literal[
        "correct",
        "mostly_correct",
        "promising_but_flawed",
        "incorrect"
    ]

    confidence_score: float   # 0.0–1.0 reviewer confidence
    time_elapsed_sec: float | None = None
