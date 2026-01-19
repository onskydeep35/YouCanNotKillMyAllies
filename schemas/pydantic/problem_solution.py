from typing import List
from pydantic import BaseModel, Field

class ProblemSolution(BaseModel):
    answer: str = Field(..., description="Final answer to the problem")
    reasoning: List[str] = Field(
        ..., description="Step-by-step reasoning leading to the answer"
    )
    time_elapsed_sec: float | None = Field(
        default=None,
        exclude=True
    )
