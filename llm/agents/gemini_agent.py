from typing import TypeVar
from llm.agents.agent import LLMAgent
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class GeminiAgent(LLMAgent):
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

        response = self.client.models.generate_content(
            model=self.config.model,
            contents=[
                {"role": "system", "parts": [{"text": system_prompt}]},
                {"role": "user", "parts": [{"text": user_prompt}]},
            ],
            config={
                **gen_kwargs,
                "response_mime_type": "application/json",
                "response_json_schema": output_model.model_json_schema(),
            },
        )

        return output_model.model_validate_json(response.text)
