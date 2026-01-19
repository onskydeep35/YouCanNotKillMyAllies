from pydantic import BaseModel, Field
from typing import Dict


class RoleAssessment(BaseModel):
    """
    LLM self-assessment of suitability for different roles.
    """
    llm_id: str | None = Field(
        default=None,
        exclude=True
    )
    role_scores: Dict[str, float] = Field(
        ..., description="Confidence scores per role (0.0–1.0)"
    )
    reasoning: str
