from pydantic import BaseModel, Field


class AnswerComparisonInput(BaseModel):
    """
    Input for binary answer correctness evaluation.
    """

    problem_statement: str = Field(
        ...,
        description="Full problem statement for context"
    )

    ground_answer: str = Field(
        ...,
        description="Ground-truth correct answer"
    )

    solution_answer: str = Field(
        ...,
        description="Answer produced by the solver or judge"
    )
