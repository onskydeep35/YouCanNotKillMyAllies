from pydantic import BaseModel, Field


class AnswerCorrectnessJudgement(BaseModel):
    """
    Binary judgement indicating whether the solution answer
    matches the ground-truth answer.
    """

    is_correct: bool = Field(
        ...,
        description="True if the solution answer is semantically identical to the ground-truth answer, otherwise false"
    )
