from abc import ABC, abstractmethod
import asyncio
import time
from typing import TypeVar

from schemas.pydantic.input.problem import Problem
from schemas.dataclass.agent_config import LLMAgentConfig
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMAgent(ABC):
    def __init__(self, *, config: LLMAgentConfig):
        self.config = config

    async def run_structured_call(
        self,
        *,
        problem: Problem,
        system_prompt: str,
        user_prompt: str,
        output_model: type[T],
        method_type: str,
        timeout_sec: int = 2000,
        log_interval_sec: int = 5,
    ) -> T:
        return await self._run_llm_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=output_model,
            timeout_sec=timeout_sec,
            log_interval_sec=log_interval_sec,
            instance_id=problem.problem_id,
            method_type=method_type,
        )

    async def _run_llm_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_model: type[T],
        timeout_sec: int,
        log_interval_sec: int,
        instance_id: str,
        method_type: str,
        post_call_delay_sec: float = 5,
    ) -> T:

        start_time = time.monotonic()

        async def log_progress():
            try:
                while True:
                    elapsed = time.monotonic() - start_time
                    print(
                        f"[thinking] agent={self.config.llm_id} "
                        f"method={method_type} "
                        f"instance={instance_id} "
                        f"elapsed={elapsed:.1f}s"
                    )
                    await asyncio.sleep(log_interval_sec)
            except asyncio.CancelledError:
                pass

        progress_task = asyncio.create_task(log_progress())

        try:
            result: T = await asyncio.wait_for(
                asyncio.to_thread(
                    self._call_provider,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_model=output_model,
                    method_type=method_type,
                    instance_id=instance_id,
                ),
                timeout=timeout_sec,
            )

            elapsed = time.monotonic() - start_time
            if hasattr(result, "time_elapsed_sec"):
                result.time_elapsed_sec = elapsed

            return result

        finally:
            progress_task.cancel()
            if post_call_delay_sec > 0:
                await asyncio.sleep(post_call_delay_sec)

    def _build_generation_kwargs(self) -> dict:
        """
        Build generic generation kwargs.
        Providers may ignore unsupported fields.
        """
        kwargs = {}

        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p

        return kwargs

    @abstractmethod
    def _call_provider(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_model: type[T],
        method_type: str,
        instance_id: str,
    ) -> T:
        """Call the LLM provider with structured output parsing.

        This method must be implemented by each provider-specific agent subclass
        to handle the provider's API and response format.

        Args:
            system_prompt: The system-level instruction for the model.
            user_prompt: The user input/query for the model.
            output_model: The Pydantic model class for structured output validation.
            method_type: The method/task type for logging and debugging.
            instance_id: A unique identifier for this invocation instance.

        Returns:
            An instance of output_model containing the parsed API response.
        """
        pass
