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
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=[
                {"role": "system", "parts": [{"text": system_prompt}]},
                {"role": "user", "parts": [{"text": user_prompt}]},
            ],
            config={
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "response_mime_type": "application/json",
                "response_json_schema": output_model.model_json_schema(),
            },
        )

        print(
            f"\n[RAW GEMINI OUTPUT] "
            f"agent={self.config.llm_id} "
            f"method={method_type} "
            f"instance={instance_id}\n"
            f"{response.text}\n"
        )

        return output_model.model_validate_json(response.text)
