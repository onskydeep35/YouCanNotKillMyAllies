from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Problem:
    id: str
    category: str
    subcategory: Optional[str]
    statement: str
    ground_answer: str
    difficulty: str

    @staticmethod
    def from_dict(data: Dict) -> "Problem":
        return Problem(
            id=data["id"],
            category=data["category"],
            subcategory=data.get("subcategory"),
            statement=data["problem_statement"],
            ground_answer=data["ground_answer"],
            difficulty=data["difficulty"]
        )

    @staticmethod
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "metadata": self.metadata,
        }