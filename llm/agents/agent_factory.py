from typing import Iterable, List

from llm.agents.agent import LLMAgent
from llm.agents.openai_agent import OpenAIAgent
from llm.agents.gemini_agent import GeminiAgent
from schemas.dataclass.agent_config import LLMAgentConfig
from llm.clients.provider_registry import ProviderClientRegistry


class AgentFactory:
    @staticmethod
    def create_agent(config: LLMAgentConfig) -> LLMAgent:
        provider = config.provider.lower()

        if provider == "openai":
            return OpenAIAgent(
                client=ProviderClientRegistry.get_openai_client(),
                config=config,
            )

        if provider == "gemini":
            return GeminiAgent(
                client=ProviderClientRegistry.get_gemini_client(),
                config=config,
            )

        raise ValueError(
            f"Unsupported LLM provider '{config.provider}' "
            f"for agent '{config.llm_id}'"
        )

    @classmethod
    def create_agents(
        cls,
        configs: Iterable[LLMAgentConfig],
    ) -> List[LLMAgent]:
        return [cls.create_agent(cfg) for cfg in configs]
