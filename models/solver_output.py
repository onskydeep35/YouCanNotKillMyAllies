from dataclasses import dataclass
from typing import List


@dataclass
class SolverOutput:
    problem_id: str
    llm_model: str
    answer: str
    reasoning: List[str]
