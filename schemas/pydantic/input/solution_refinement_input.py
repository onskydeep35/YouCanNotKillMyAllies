from typing import List
from pydantic import BaseModel, Field
from schemas.pydantic.input.problem import Problem
from schemas.pydantic.output.problem_solution import ProblemSolution
from schemas.pydantic.output.problem_solution_review import ProblemSolutionReview

class SolutionRefinementInput(BaseModel):
    problem: Problem = Field(
        ...,
        description="The authoritative problem definition that the solution and reviews are based on."
    )

    solution: ProblemSolution = Field(
        ...,
        description="The original solver-produced solution that is subject to refinement."
    )

    reviews: List[ProblemSolutionReview] = Field(
        ...,
        description="Peer review feedback that must be addressed during solution refinement."
    )
