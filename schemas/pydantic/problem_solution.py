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
    answer: str = Field(..., description="Final answer to the problem")
    reasoning: List[str] = Field(
        ..., description="Step-by-step reasoning leading to the answer"
    )
