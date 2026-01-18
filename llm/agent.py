import asyncio
import time
from typing import Type, TypeVar

from llm.models.problem import Problem
from llm.models.solver_output import SolverOutput
from llm.prompts import build_solver_user_prompt
from llm.models.agent_config import *
from llm.models.role_assessment import *

T = TypeVar("T")  # for structured LLM outputs


class LLMAgent:
    """
    Executes LLM calls using a fixed LLMAgentConfig.
    """

    def __init__(self, *, client, config: LLMAgentConfig):
        self.client = client
        self.config = config

    # -------------------------
    # Internal helper
    # -------------------------
    async def _run_llm_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_model: Type[T],
        timeout_sec: int,
        log_interval_sec: int,
        problem_id: str,
    ) -> T:
        start_time = time.monotonic()

        async def log_progress():
            try:
                while True:
                    elapsed = time.monotonic() - start_time
                    print(
                        f"[thinking] agent={self.config.llm_id} "
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
                    # 🔒 ALWAYS from config
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "response_mime_type": "application/json",
                    "response_json_schema": output_model.model_json_schema(),
                },
            )

            print(
                f"\n[RAW LLM OUTPUT] agent={self.config.llm_id} "
                f"problem={problem_id}\n{response.text}\n"
            )

            return output_model.model_validate_json(response.text)

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_call_sync),
                timeout=timeout_sec,
            )

        finally:
            progress_task.cancel()

    async def assess_roles_for_problem(
            self,
            *,
            problem: Problem,
            timeout_sec: int = 300,
            log_interval_sec: int = 5,
    ) -> RoleAssessment:
        """
        Phase 0: LLM self-assesses suitability for each role.
        """

        system_prompt = (
            "You are part of a multi-agent reasoning system.\n\n"
            "Your task is to assess your suitability for each role below:\n"
            "- Solver: independently solves the problem.\n"
            "- Judge: evaluates and critiques solutions from others.\n\n"
            "For the given problem, estimate your suitability for EACH role.\n"
            "Return confidence scores between 0.0 and 1.0.\n"
            "Do NOT choose a final role."
        )

        user_prompt = (
            f"Problem:\n{problem.statement}\n\n"
            "Assess your suitability for each role."
        )

        output = await self._run_llm_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=RoleAssessment,
            timeout_sec=timeout_sec,
            log_interval_sec=log_interval_sec,
            problem_id=problem.id,
        )

        # 🔒 Deterministically inject agent identity
        output.llm_id = self.config.llm_id
        return output

    # -------------------------
    # Public API: Solver
    # -------------------------
    async def solve_problem(
        self,
        *,
        problem: Problem,
        system_prompt: str,
        timeout_sec: int = 2000,
        log_interval_sec: int = 5,
    ) -> SolverOutput:
        """
        Solve a problem using the given system prompt.
        """
        user_prompt = build_solver_user_prompt(problem)

        try:
            return await self._run_llm_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_model=SolverOutput,
                timeout_sec=timeout_sec,
                log_interval_sec=log_interval_sec,
                problem_id=problem.id,
            )

        except asyncio.TimeoutError:
            print(
                f"\n[TIMEOUT] agent={self.config.llm_id} "
                f"problem={problem.id} after {timeout_sec}s\n"
            )

            return SolverOutput(
                problem_id=problem.id,
                llm_model=self.config.model,
                answer="TIMEOUT",
                reasoning=[
                    "Solver timed out before producing an answer."
                ],
            )
