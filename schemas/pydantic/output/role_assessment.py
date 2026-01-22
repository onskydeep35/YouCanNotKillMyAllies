from pydantic import BaseModel, Field

class RoleAssessment(BaseModel):
    """
    Self-assessment produced by an LLM indicating its suitability for different roles.
    Used during Stage 0 and Stage 0.5 for algorithmic role assignment.
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

    assessment_id: str | None = Field(
        default=None,
        exclude=True,
        description="Internal identifier for this role assessment instance (excluded from model output)"
    )

    problem_id: str | None = Field(
        default=None,
        exclude=True,
        description="Identifier of the problem for this assessment (excluded from model output)"
    )

    judge_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM's score for being a judge for given problem (0–1)"
    )

    solver_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM's score for being a solver for given problem (0–1)"
    )

    reasoning: str = Field(
        ...,
        description="Explanation justifying why the LLM believes it is suitable for the given roles"
    )
