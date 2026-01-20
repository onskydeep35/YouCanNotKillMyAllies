from pydantic import BaseModel, Field


class RoleScore(BaseModel):
    """
    Confidence score assigned by the LLM for a specific role.
    """

    role: str = Field(
        ...,
        description="Name of the role being evaluated (e.g., 'Solver_1', 'Solver_2', 'Solver_3', 'Judge')"
    )

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM's self-assessed suitability for this role on a scale from 0 to 1"
    )


class RoleAssessment(BaseModel):
    """
    Self-assessment produced by an LLM indicating its suitability for different roles.
    Used during Stage 0 and Stage 0.5 for algorithmic role assignment.
    """

    llm_id: str | None = Field(
        default=None,
        exclude=True,
        description="Identifier of the LLM producing this assessment (excluded from model output)"
    )

    assessment_id: str | None = Field(
        default=None,
        exclude=True,
        description="Internal identifier for this role assessment instance"
    )

    role_scores: list[RoleScore] = Field(
        ...,
        description="List of roles with corresponding self-assessed suitability scores"
    )

    reasoning: str = Field(
        ...,
        description="Explanation justifying why the LLM believes it is suitable for the given roles"
    )
