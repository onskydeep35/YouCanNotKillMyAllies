import asyncio
from typing import List

from llm.agent import LLMAgent
from llm.models.dataclass.problem import Problem
from pipeline.run_context import RunContext


class RoleDeterminationStage:
    def __init__(self, agents: List[LLMAgent], max_concurrency: int = 4):
        self.agents = agents
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def run(
        self,
        *,
        ctx: RunContext,
        problems: List[Problem],
        timeout_sec: int,
        log_interval_sec: int,
    ) -> None:
        """
        - Collect role self-assessments (ephemeral)
        - Deterministically assign final roles
        """

        if not problems:
            raise RuntimeError("No problems provided")

        reference_problem = problems[0]

        async def assess(agent: LLMAgent):
            async with self.semaphore:
                return await agent.assess_roles_for_problem(
                    problem=reference_problem,
                    timeout_sec=timeout_sec,
                    log_interval_sec=log_interval_sec,
                )

        results = await asyncio.gather(
            *[assess(agent) for agent in self.agents],
            return_exceptions=True,
        )

        valid_assessments = []
        for agent, result in zip(self.agents, results):
            if isinstance(result, Exception):
                print(
                    f"[ROLE FAIL] agent={agent.config.llm_id} error={result}"
                )
                continue
            valid_assessments.append(result)

        if not valid_assessments:
            raise RuntimeError("All role self-assessments failed")

        self._assign_roles(ctx, valid_assessments)

    def _assign_roles(self, ctx: RunContext, assessments) -> None:
        sorted_agents = sorted(
            assessments,
            key=lambda a: (
                a.role_scores.get("Solver", 0.0),
                a.llm_id,
            ),
            reverse=True,
        )

        num_solvers = (len(sorted_agents) + 1) // 2

        for idx, assessment in enumerate(sorted_agents):
            ctx.final_roles[assessment.llm_id] = (
                "Solver" if idx < num_solvers else "Judge"
            )
