from dataclasses import dataclass
from typing import Optional


@dataclass
class JudgeOutput:
    problem_id: str
    selected_llm: str
    justification: str
    confidence: Optional[float]
