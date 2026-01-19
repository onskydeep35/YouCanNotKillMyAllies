import asyncio
import time
from typing import Type, TypeVar

from llm.models.dataclass.problem import Problem
from llm.models.dataclass.agent_config import LLMAgentConfig

T = TypeVar("T")


class LLMAgent:
    """
    Generic LLM execution engine.
    Executes schema-constrained LLM calls.
    """

    def __init__(self, *, client, config: LLMAgentConfig):
        self.client = client
        self.config = config


    async def run_structured_call(
            self,
            *,
            problem: Problem,
            system_prompt: str,
            user_prompt: str,
            output_model: Type[T],
            method_type: str,
            timeout_sec: int = 2000,
            log_interval_sec: int = 5,
    ) -> T:
        """
        Execute a structured LLM call and return a validated output model.
        """

        try:
            return await self._run_llm_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_model=output_model,
                timeout_sec=timeout_sec,
                log_interval_sec=log_interval_sec,
                problem_id=problem.id,
                method_type=method_type,
            )

        except asyncio.TimeoutError:
            print(
                f"\n[TIMEOUT] "
                f"agent={self.config.llm_id} "
                f"method={method_type} "
                f"problem={problem.id} "
                f"after {timeout_sec}s\n"
            )
            raise

    async def _run_llm_call(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            output_model: Type[T],
            timeout_sec: int,
            log_interval_sec: int,
            problem_id: str,
            method_type: str,
            post_call_delay_sec: float = 5,
    ) -> T:
        start_time = time.monotonic()
        result: T | None = None

        async def log_progress():
            try:
                while True:
                    elapsed = time.monotonic() - start_time
                    print(
                        f"[thinking] "
                        f"agent={self.config.llm_id} "
                        f"method={method_type} "
                        f"problem={problem_id} "
                        f"elapsed={elapsed:.1f}s"
                    )
                    await asyncio.sleep(log_interval_sec)
            except asyncio.CancelledError:
                pass

        progress_task = asyncio.create_task(log_progress())

        def _call_sync() -> T:
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
                f"\n[RAW LLM OUTPUT] "
                f"agent={self.config.llm_id} "
                f"method={method_type} "
                f"problem={problem_id}\n"
                f"{response.text}\n"
            )

            return output_model.model_validate_json(response.text)

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_call_sync),
                timeout=timeout_sec,
            )

        finally:
            # stop logger
            progress_task.cancel()

            # inject elapsed time if result exists
            if result is not None:
                elapsed = time.monotonic() - start_time
                if hasattr(result, "time_elapsed_sec"):
                    result.time_elapsed_sec = elapsed

            # unconditional delay
            if post_call_delay_sec > 0:
                await asyncio.sleep(post_call_delay_sec)

        return result


