from pydantic import BaseModel

class RoleScore(BaseModel):
    role: str
    score: float

class RoleAssessment(BaseModel):
    llm_id: str
    role_scores: list[RoleScore]
    reasoning: str
