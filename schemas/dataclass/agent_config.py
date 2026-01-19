from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMAgentConfig:
    provider: str
    llm_id: str
    model: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
