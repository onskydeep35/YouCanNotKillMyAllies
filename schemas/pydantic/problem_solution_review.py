from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ReviewError(BaseModel):
    """
    Represents a concrete error identified in a peer's solution.
    """

    location: str = Field(
        ...,
        description="Where the error occurs in the solution (e.g., 'Step 5', 'Initial assumption', 'Case n=0')"
    )

    error_type: Literal[
        "logical_error",
        "missing_case",
        "invalid_assumption",
        "math_error",
        "inconsistency",
        "unclear_reasoning"
    ] = Field(
        ...,
        description="Category of the error detected in the solution"
    )

    description: str = Field(
        ...,
        description="Detailed explanation of why this is an error"
    )

    severity: Literal["minor", "major", "critical"] = Field(
        ...,
        description="Impact level of the error on the solution's correctness"
    )


class PeerEvaluation(BaseModel):
    """
    Structured qualitative and quantitative evaluation of a peer's solution.
    """

    strengths: List[str] = Field(
        ...,
        description="Aspects of the solution that are correct, clear, or well-reasoned"
    )

    weaknesses: List[str] = Field(
        ...,
        description="Identified weaknesses that reduce clarity, rigor, or correctness"
    )

    errors: List[ReviewError] = Field(
        ...,
        description="Explicitly identified errors found in the solution"
    )

    suggested_changes: List[str] = Field(
        ...,
        description="Concrete suggestions for improving or fixing the solution"
    )


class ProblemSolutionReview(BaseModel):
    """
    Complete peer review of a solver's solution produced during Stage 2.
    """

    review_id: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Internal identifier for the review (excluded from model output)"
    )

    run_id: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Identifier for the system run or experiment instance"
    )

    solution_id: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Identifier of the reviewed solution"
    )

    problem_id: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Identifier of the problem being reviewed"
    )

    reviewer_id: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Identifier of the solver acting as the reviewer"
    )

    reviewee_id: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Identifier of the solver whose solution is being reviewed"
    )

    time_elapsed_sec: Optional[float] = Field(
        default=None,
        exclude=True,
        description="Time taken to produce this review (in seconds)"
    )

    evaluation: PeerEvaluation = Field(
        ...,
        description="Structured evaluation containing strengths, weaknesses, errors, and suggestions"
    )

    overall_assessment: Literal[
        "correct",
        "mostly_correct",
        "promising_but_flawed",
        "incorrect"
    ] = Field(
        ...,
        description="High-level judgment of the solution's correctness"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Reviewer's confidence in the overall assessment (0–1)"
    )
