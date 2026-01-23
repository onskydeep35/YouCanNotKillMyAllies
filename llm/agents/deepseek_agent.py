from typing import TypeVar
from openai import OpenAI
from pydantic import BaseModel
from llm.agents.agent import LLMAgent

T = TypeVar("T", bound=BaseModel)


class DeepSeekAgent(LLMAgent):
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

        response = self.client.beta.chat.completions.parse(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=output_model,
            **gen_kwargs,
        )

        print(
            f"\n[RAW DEEPSEEK OUTPUT] "
            f"agent={self.config.llm_id} "
            f"method={method_type} "
            f"instance={instance_id}\n"
            f"{response.choices[0].message.content}\n"
        )

        return response.choices[0].message.parsed