from typing import List, Literal
from pydantic import BaseModel, Field


class SolutionError(BaseModel):
    location: str = Field(
        ..., description="Where the error occurs (e.g., Step 5, assumption, final answer)"
    )
    error_type: Literal[
        "logical_error",
        "calculation_error",
        "unsupported_claim",
        "missing_case",
        "ambiguity",
        "other",
    ]
    description: str = Field(
        ..., description="Explanation of why this is an error"
    )
    severity: Literal["minor", "major", "critical"]


class ProblemSolutionJudgement(BaseModel):
    solution_id: str = Field(
        ..., description="Identifier of the solution being reviewed"
    )

    strengths: List[str] = Field(
        ..., description="What the solution does well"
    )

    weaknesses: List[str] = Field(
        ..., description="Problems or limitations in the solution"
    )

    errors: List[SolutionError] = Field(
        ..., description="Concrete errors found in the solution"
    )

    suggested_changes: List[str] = Field(
        ..., description="Actionable suggestions for improvement"
    )

    overall_assessment: Literal[
        "correct",
        "mostly_correct",
        "promising_but_flawed",
        "incorrect",
    ] = Field(
        ..., description="Overall quality assessment of the solution"
    )