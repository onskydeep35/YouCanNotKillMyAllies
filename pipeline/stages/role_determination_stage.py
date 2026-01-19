import asyncio
from typing import List

from llm.agents.agent import LLMAgent
from schemas.dataclass.problem import Problem
from schemas.pydantic.role_assessment import RoleAssessment
from pipeline.run_context import RunContext
from llm.prompts import (
    ROLE_DETERMINATION_SYSTEM_PROMPT,
    build_role_determination_user_prompt,
)


class RoleDeterminationStage:
    """
    Stage 0: Role determination for a single problem.
    """

    def __init__(
        self,
        agents: List[LLMAgent],
        *,
        max_concurrency: int = 4,
    ):
        self.agents = agents
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def run(
        self,
        *,
        ctx: RunContext,
        problem: Problem,
        timeout_sec: int,
        log_interval_sec: int,
    ) -> None:

        async def assess(agent: LLMAgent):
            async with self.semaphore:
                assessment = await agent.run_structured_call(
                    problem=problem,
                    system_prompt=ROLE_DETERMINATION_SYSTEM_PROMPT,
                    user_prompt=build_role_determination_user_prompt(problem),
                    output_model=RoleAssessment,
                    method_type="role_determination",
                    timeout_sec=timeout_sec,
                    log_interval_sec=log_interval_sec,
                )

                # Enforce agent identity
                assessment.llm_id = agent.config.llm_id
                return assessment

        results = await asyncio.gather(
            *[assess(agent) for agent in self.agents],
            return_exceptions=True,
        )

        valid_assessments: List[RoleAssessment] = []
        for agent, result in zip(self.agents, results):
            if isinstance(result, Exception):
                print(
                    f"[ROLE FAIL] "
                    f"problem={problem.id} "
                    f"agent={agent.config.llm_id} "
                    f"error={result}"
                )
                continue
            valid_assessments.append(result)

        if not valid_assessments:
            raise RuntimeError(
                f"All role self-assessments failed for problem {problem.id}"
            )

        self._assign_roles(ctx, valid_assessments)

    def _assign_roles(
        self,
        ctx: RunContext,
        assessments: List[RoleAssessment],
    ) -> None:
        """
        Deterministic role assignment for this problem.
        """

        def solver_score(a: RoleAssessment) -> float:
            for rs in a.role_scores:
                if rs.role == "Solver":
                    return rs.score
            return 0.

        sorted_assessments = sorted(
            assessments,
            key=lambda a: solver_score(a),
            reverse=True,
        )

        num_solvers = len(sorted_assessments) - 1

        for idx, assessment in enumerate(sorted_assessments):
            ctx.final_roles[assessment.llm_id] = (
                "Solver" if idx < num_solvers else "Judge"
            )
