from typing import List
from pydantic import BaseModel, Field
from schemas.pydantic.input.problem import Problem
from schemas.pydantic.output.problem_solution import ProblemSolution
from schemas.pydantic.output.problem_solution_review import ProblemSolutionReview
from schemas.pydantic.output.refined_problem_solution import RefinedProblemSolution

class SolverContexts(BaseModel):
    solution: ProblemSolution = Field(
        ...,
        description="The original solution produced by a solver before peer review and refinement."
    )

    received_reviews: List[ProblemSolutionReview] = Field(
        ...,
        description="All peer review feedback received for this solver’s original solution."
    )

    refined_solution: RefinedProblemSolution = Field(
        ...,
        description="The solver’s final refined solution after addressing peer review feedback."
    )


class FinalJudgementInput(BaseModel):
    problem: Problem = Field(
        ...,
        description="The authoritative problem definition used to evaluate all solver solutions."
    )

    solver_contexts: List[SolverContexts] = Field(
        ...,
        description="Per-solver solution contexts containing original solutions, reviews, and refined results for comparison."
    )
