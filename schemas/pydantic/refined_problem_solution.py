from typing import List, Optional
from pydantic import BaseModel, Field


class CritiqueResolution(BaseModel):
    """
    Represents how the solver handled a single peer critique.
    """
    critique_id: Optional[str] = Field(
        None,
        description="Optional identifier linking back to a specific peer review critique"
    )

    critique: str = Field(
        ...,
        description="The critique raised by a peer reviewer"
    )

    response: str = Field(
        ...,
        description="Solver's response explaining how the critique was addressed or rebutted"
    )

    accepted: bool = Field(
        ...,
        description="Whether the critique was accepted and incorporated into the solution"
    )


class RefinedProblemSolution(BaseModel):
    refined_solution_id: str | None = Field(
        default=None,
        exclude=True
    )

    parent_solution_id: str | None = Field(
        default=None,
        exclude=True
    )

    run_id: str | None = Field(
        default=None,
        exclude=True
    )

    solver_llm_model_id: str | None = Field(
        default=None,
        exclude=True
    )

    problem_id: str | None = Field(
        default=None,
        exclude=True
    )

    review_ids: List[str] | None = Field(
        default=None,
        exclude=True
    )

    time_elapsed_sec: Optional[float] = Field(
        default=None,
        exclude=True,
        description="Time taken to produce this review (in seconds)"
    )

    changes_made: List[CritiqueResolution] = Field(
        ...,
        description="List of critique resolutions addressing all peer feedback"
    )

    refined_reasoning: List[str] = Field(
        ...,
        description="Full revised reasoning (steps to solution) after incorporating valid feedback"
    )

    refined_answer: str = Field(
        ...,
        description="Final concise answer to the problem"
    )

    answer_changed: bool = Field(
        ...,
        description="Whether the initial and refined answers changed after solution refinement"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Solver's confidence in the correctness of the refined answer (0–1)"
    )
