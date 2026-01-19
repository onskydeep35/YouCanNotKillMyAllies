from dataclasses import dataclass


@dataclass(frozen=True)
class LLMAgentConfig:
    """
    Defines the identity and decoding behavior of an LLM agent.
    """
    provider: str
    llm_id: str
    model: str
    temperature: float
    top_p: float
