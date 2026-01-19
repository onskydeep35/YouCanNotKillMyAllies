import asyncio
from typing import List

from llm.agent import LLMAgent
from llm.models.dataclass.problem import Problem
from llm.models.dataclass.roles import LLMRolePreference
from llm.prompts import build_system_prompt
from pipeline.run_context import RunContext

from data.firestore_writer import FirestoreWriter
from data.firestore_writer import SOLUTIONS


class SolverStage:
    def __init__(
        self,
        agents: List[LLMAgent],
        *,
        writer: FirestoreWriter,
        max_concurrency: int = 4,
    ):
        self.agents = agents
        self.writer = writer
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def run(
        self,
        *,
        ctx: RunContext,
        problems: List[Problem],
        timeout_sec: int,
        log_interval_sec: int,
    ) -> None:

        solver_agents = [
            a for a in self.agents
            if ctx.final_roles.get(a.config.llm_id) == "Solver"
        ]

        if not solver_agents:
            raise RuntimeError("No solver agents assigned")

        solver_role = LLMRolePreference(
            role_preferences=["Solver"],
            confidence_by_role={"Solver": 1.0},
            reasoning="Solve the problem using precise logical reasoning.",
        )

        system_prompt = build_system_prompt(solver_role.reasoning)

        async def solve(agent: LLMAgent, problem: Problem):
            async with self.semaphore:
                solution = await agent.solve_problem(
                    problem=problem,
                    system_prompt=system_prompt,
                    timeout_sec=timeout_sec,
                    log_interval_sec=log_interval_sec,
                )

                # 🔥 Firestore write happens HERE
                await self.writer.write(
                    collection=SOLUTIONS,
                    document={
                        "run_id": ctx.run_id,
                        "problem_id": problem.id,
                        "solver_id": agent.config.llm_id,
                        "model": agent.config.model,
                        "temperature": agent.config.temperature,
                        "top_p": agent.config.top_p,
                        **solution.model_dump(),
                    },
                )

                return solution

        tasks = [(a, p) for a in solver_agents for p in problems]
        results = await asyncio.gather(
            *[solve(a, p) for a, p in tasks],
            return_exceptions=True,
        )

        for (agent, problem), result in zip(tasks, results):
            if isinstance(result, Exception):
                print(
                    f"[FAIL] solver={agent.config.llm_id} "
                    f"problem={problem.id} error={result}"
                )
            else:
                print(
                    f"[OK] run={ctx.run_id} "
                    f"problem={problem.id} solver={agent.config.llm_id}"
                )
