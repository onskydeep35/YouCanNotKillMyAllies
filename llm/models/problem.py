from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Problem:
    id: str
    category: str
    subcategory: Optional[str]
    statement: str
    query: str
    answer: str
    full_assignment: Optional[Dict]

    @staticmethod
    def from_dict(data: Dict) -> "Problem":
        return Problem(
            id=data["id"],
            category=data["category"],
            subcategory=data.get("subcategory"),
            statement=data["problem_statement"],
            query=data["ground_truth"]["query"],
            answer=data["ground_truth"]["answer"],
            full_assignment=data["ground_truth"].get("full_assignment"),
        )

    @staticmethod
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "metadata": self.metadata,
        }