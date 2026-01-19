from pydantic import BaseModel, Field

class RoleScore(BaseModel):
    role: str
    score: float

class RoleAssessment(BaseModel):
    llm_id: str | None = Field(
        default=None,
        exclude=True
    )
    assessment_id: str | None = Field(
        default=None,
        exclude=True
    )
    role_scores: list[RoleScore]
    reasoning: str
