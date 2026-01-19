from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class SolutionError(BaseModel):
    location: Optional[str] = Field(
        None,
        description="Where the error occurs (e.g., 'Step 5', 'Final conclusion')"
    )
    error_type: Literal[
        "logical_error",
        "calculation_error",
        "missing_case",
        "unsupported_claim",
        "ambiguity",
        "other"
    ]
    description: str = Field(
        ...,
        description="Clear explanation of why this is an error"
    )
    severity: Literal["minor", "major", "critical"]


class SolutionEvaluation(BaseModel):
    strengths: List[str] = Field(
        default_factory=list,
        description="Correct or well-explained aspects of the solution"
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Non-fatal issues, unclear steps, or missing explanations"
    )
    errors: List[SolutionError] = Field(
        default_factory=list,
        description="Concrete errors found in the solution"
    )
    suggested_changes: List[str] = Field(
        default_factory=list,
        description="Actionable suggestions to improve the solution"
    )


class ProblemSolutionJudgement(BaseModel):
    solution_id: str = Field(
        ...,
        description="Identifier of the solution being reviewed (solver id)"
    )

    evaluation: SolutionEvaluation

    overall_assessment: Literal[
        "correct",
        "mostly_correct",
        "promising_but_flawed",
        "incorrect"
    ] = Field(
        ...,
        description="High-level judgement of solution quality"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Reviewer confidence in this judgement"
    )
