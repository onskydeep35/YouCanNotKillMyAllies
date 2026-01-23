from typing import TypeVar
from openai import OpenAI
from pydantic import BaseModel
from llm.agents.agent import LLMAgent

T = TypeVar("T", bound=BaseModel)


class OpenAIAgent(LLMAgent):
    def __init__(self, *, client, config):
        super().__init__(config=config)
        self.client = client

    def _call_provider(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_model: type[T],
        method_type: str,
        instance_id: str,
    ) -> T:
        gen_kwargs = self._build_generation_kwargs()

        response = self.client.responses.parse(
            model=self.config.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=output_model,
            **gen_kwargs,  # ← generic kwargs
        )

        print(
            f"\n[RAW OPENAI OUTPUT] "
            f"agent={self.config.llm_id} "
            f"method={method_type} "
            f"instance={instance_id}\n"
            f"{response.output_text}\n"
        )

        return response.output_parsed
