from typing import Dict, Optional
from pydantic import BaseModel, Field


class Problem(BaseModel):
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

    problem_id: str = Field(
        default=None,
        exclude=True,
        description="Unique identifier of the problem"
    )

    category: str = Field(
        ...,
        description="High-level category of the problem (e.g., logic, math, physics)"
    )

    subcategory: Optional[str] = Field(
        None,
        description="Optional subcategory of the problem"
    )

    statement: str = Field(
        ...,
        description="Full problem statement presented to the solver"
    )

    ground_answer: str = Field(
        default=None,
        exclude=True,
        description="Ground-truth correct answer used for evaluation"
    )

    difficulty: str = Field(
        default=None,
        exclude=True,
        description="Difficulty level of the problem (e.g., easy, medium, hard)"
    )

    @classmethod
    def from_dict(cls, data: Dict) -> "Problem":
        """
        Construct a Problem from a raw dictionary (e.g., dataset JSON).
        """
        return cls(
            problem_id=data["id"],
            category=data["category"],
            subcategory=data.get("subcategory"),
            statement=data["problem_statement"],
            ground_answer=data["ground_answer"],
            difficulty=data["difficulty"],
        )

    def to_dict(self) -> Dict:
        """
        Serialize the Problem into a dictionary for storage or transport.
        """
        return {
            "id": self.problem_id,
            "category": self.category,
            "subcategory": self.subcategory,
            "problem_statement": self.statement,
            "ground_answer": self.ground_answer,
            "difficulty": self.difficulty,
        }
