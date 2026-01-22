from typing import Dict, Optional
from pydantic import BaseModel, Field
from schemas.pydantic.input.problem import Problem
from schemas.pydantic.output.problem_solution import ProblemSolution

class ReviewInput(BaseModel):
    Problem: Problem = Field(
        ...,
        description="Problem given for this review"
    )

    ProblemSolution: ProblemSolution = Field(
        ...,
        description="Problem solution given for this review"
    )
