from typing import Dict, Optional
from pydantic import BaseModel, Field
from schemas.pydantic.input.problem import Problem
from schemas.pydantic.output.problem_solution import ProblemSolution

class PeerReviewInput(BaseModel):
    problem: Problem = Field(
        ...,
        description="Problem given for this review"
    )

    solution: ProblemSolution = Field(
        ...,
        description="Solution for the problem given for this review"
    )
