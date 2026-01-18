from typing import List
from pydantic import BaseModel, Field


class SolverOutput(BaseModel):
    answer: str = Field(..., description="Final answer to the problem")
    reasoning: List[str] = Field(
        ..., description="Step-by-step reasoning leading to the answer"
    )
