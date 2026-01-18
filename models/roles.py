from dataclasses import dataclass
from typing import Dict, List


@dataclass
class LLMRolePreference:
    role_preferences: List[str]
    confidence_by_role: Dict[str, float]
    reasoning: str
