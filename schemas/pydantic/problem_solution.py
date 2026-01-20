from typing import List
from pydantic import BaseModel, Field

class ProblemSolution(BaseModel):
    solution_id: str | None = Field(
        default=None,
        exclude=True
    )

    problem_id: str | None = Field(
        default=None,
        exclude=True
    )

    solver_llm_model_id: str | None = Field(
        default=None,
        exclude=True
    )

    run_id: str | None = Field(
        default=None,
        exclude=True
    )

    time_elapsed_sec: float | None = Field(
        default=None,
        exclude=True
    )

    reasoning: List[str] = Field(
        ..., description="Step-by-step reasoning leading to the answer"
    )

    answer: str = Field(..., description="Final answer to the problem")

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Solver's confidence in the correctness of the answer (0–1)"
    )
