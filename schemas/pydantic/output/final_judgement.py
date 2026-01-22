from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class FinalJudgement(BaseModel):
    """
    Output schema for Stage 4: Final Judgment.

    Produced by the Judge after evaluating:
    - All original solutions
    - All peer reviews
    - All refined solutions
    """

    prompt_system: str | None = Field(
        default=None,
        exclude=True,
        description="System Prompt for LLM call (excluded from model output)"
    )

    prompt_user: str | None = Field(
        default=None,
        exclude=True,
        description="User Prompt for LLM call (excluded from model output)"
    )

    llm_id: str | None = Field(
        default=None,
        exclude=True,
        description="Identifier of the LLM producing this assessment (excluded from model output)"
    )

    run_id: str | None = Field(
        default=None,
        exclude=True,
        description="Identifier of the problem solving session run (excluded from model output)"
    )

    judgement_id: str | None = Field(
        default=None,
        exclude=True,
        description="Internal identifier for this role assessment instance (excluded from model output)"
    )

    problem_id: str | None = Field(
        default=None,
        exclude=True,
        description="Identifier of the problem for this assessment (excluded from model output)"
    )

    reasoning: List[str] = Field(
        ..., description="Step-by-step reasoning leading to the answer"
    )

    winner_solver: Literal["Solver 1", "Solver 2", "Solver 3"] = Field(
        ...,
        description="The most accurate and the best solution author amongst given solver agent contexts for the problem"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Judge confidence in the decision (0–1)"
    )

    time_elapsed_sec: Optional[float] = Field(
        default=None,
        exclude=True,
        description="Time taken to produce this judgement (in seconds)"
    )
