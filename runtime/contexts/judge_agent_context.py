import uuid
from typing import List, Optional
import json

from llm.agents.agent import LLMAgent
from schemas.pydantic.input.problem import Problem
from schemas.pydantic.output.final_judgement import FinalJudgement

from runtime.contexts.solver_agent_context import SolverAgentContext
from llm.prompts.prompts import (
    FINAL_JUDGMENT_SYSTEM_PROMPT,
)

class JudgeAgentContext:
    """
    Holds judge-related state and behavior for Stage 4: Final Judgment.

    The Judge evaluates:
    - The original problem
    - All solver original solutions
    - All peer reviews
    - All refined solutions

    And selects exactly one winning solver.
    """

    def __init__(
        self,
        *,
        agent: LLMAgent,
        problem: Problem,
        run_id: str,
    ):
        self.agent = agent
        self.problem = problem
        self.run_id = run_id

        self.judgment: Optional[FinalJudgement] = None

    @property
    def judge_id(self) -> str:
        return self.agent.config.llm_id

    @staticmethod
    def build_final_judgment_user_prompt(
            *,
            problem: Problem,
            solver_contexts: List[SolverAgentContext],
    ) -> str:
        """
        Builds the user prompt for Stage 4 final judgment.

        Input is structured, explicit, and sectioned to ensure
        accurate evaluation under high token load.
        """

        if len(solver_contexts) != 3:
            raise ValueError(
                "Final judgment requires exactly 3 solver contexts"
            )

        # --- Problem section ---
        problem_payload = problem.model_dump()

        solver_sections = []

        for idx, ctx in enumerate(solver_contexts, start=1):
            if ctx.solution is None:
                raise ValueError(
                    f"Solver {idx} has no original solution"
                )

            if ctx.refined_solution is None:
                raise ValueError(
                    f"Solver {idx} has no refined solution"
                )

            solver_payload = {
                "solver_id": ctx.solver_id,
                "original_solution": ctx.solution.model_dump(),
                "peer_reviews_received": [
                    review.model_dump()
                    for review in ctx.peer_reviews
                ],
                "refined_solution": ctx.refined_solution.model_dump(),
            }

            solver_sections.append(
                f"SOLVER {idx}\n{json.dumps(solver_payload, indent=2, ensure_ascii=False)}"
            )

        user_prompt = (
            "Below is the complete input for FINAL JUDGMENT.\n"
            "All information is provided as structured JSON per entity.\n"
            "You must evaluate each solver independently and select exactly one winner.\n\n"
            "PROBLEM\n"
            f"{json.dumps(problem_payload, indent=2, ensure_ascii=False)}\n\n"
            f"{solver_sections[0]}\n\n"
            f"{solver_sections[1]}\n\n"
            f"{solver_sections[2]}"
        )

        print(f"[FINAL JUDGMENT USER PROMPT]: {user_prompt}")

        return user_prompt

    async def generate_judgement(
        self,
        *,
        solver_agent_contexts: List[SolverAgentContext],
        timeout_sec: int,
        log_interval_sec: int,
    ) -> FinalJudgement:
        """
        Executes Stage 4: Final Judgement.

        Returns a fully populated FinalJudgement object.
        """

        # --- Safety checks ---
        for idx, ctx in enumerate(solver_agent_contexts, start=1):
            if ctx.solution is None:
                raise RuntimeError(
                    f"Solver {idx} has no original solution"
                )
            if ctx.refined_solution is None:
                raise RuntimeError(
                    f"Solver {idx} has no refined solution"
                )

        system_prompt = FINAL_JUDGMENT_SYSTEM_PROMPT
        user_prompt = self.build_final_judgment_user_prompt(
            problem=self.problem,
            solver_contexts=solver_agent_contexts,
        )

        judgement = await self.agent.run_structured_call(
            problem=self.problem,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=FinalJudgement,
            method_type="final_judgment",
            timeout_sec=timeout_sec,
            log_interval_sec=log_interval_sec,
        )

        judgement.prompt_system = system_prompt
        judgement.prompt_user = user_prompt
        judgement.llm_id = self.judge_id
        judgement.run_id = self.run_id
        judgement.problem_id = self.problem.problem_id
        judgement.judgement_id = uuid.uuid4().hex
        judgement.time_elapsed_sec = judgement.time_elapsed_sec

        self.judgment = judgement
        return judgement

