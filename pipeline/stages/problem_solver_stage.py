import asyncio
from typing import List
import json

from llm.agent import LLMAgent
from llm.models.dataclass.problem import Problem
from llm.models.pydantic.problem_solution import ProblemSolution
from llm.prompts import (
    build_solver_system_prompt,
    build_solver_user_prompt,
)
from pipeline.run_context import RunContext
from data.firestore_writer import FirestoreWriter, SOLUTIONS


class SolverStage:
    """
    Stage 1: Independent solution generation for a single problem.
    """

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
        problem: Problem,
        timeout_sec: int,
        log_interval_sec: int,
    ) -> None:

        solver_agents = [
            agent for agent in self.agents
            if ctx.final_roles.get(agent.config.llm_id) == "Solver"
        ]

        if not solver_agents:
            raise RuntimeError(
                f"No solver agents assigned for problem {problem.id}"
            )

        async def solve(agent: LLMAgent):
            async with self.semaphore:
                solution = await agent.run_structured_call(
                    problem=problem,
                    system_prompt=build_solver_system_prompt(
                        category=problem.category
                    ),
                    user_prompt=build_solver_user_prompt(problem),
                    output_model=ProblemSolution,
                    method_type="solver",
                    timeout_sec=timeout_sec,
                    log_interval_sec=log_interval_sec,
                )

                await self.writer.write(
                    collection=SOLUTIONS,
                    document={
                        "run_id": ctx.run_id,
                        "problem_id": problem.id,
                        "solver_id": agent.config.llm_id,
                        "model": agent.config.model,
                        "temperature": agent.config.temperature,
                        "top_p": agent.config.top_p,
                        "category": problem.category,
                        **solution.model_dump(),
                    },
                )
                
                file_path = (
                        self.output_dir /
                        f"{agent.config.llm_id}_{problem.id}.json"
                )

                file_path.write_text(
                    json.dumps(
                        solution.model_dump(),
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                return solution

        results = await asyncio.gather(
            *[solve(agent) for agent in solver_agents],
            return_exceptions=True,
        )

        for agent, result in zip(solver_agents, results):
            if isinstance(result, Exception):
                print(
                    f"[SOLVER FAIL] "
                    f"problem={problem.id} "
                    f"solver={agent.config.llm_id} "
                    f"error={result}"
                )
            else:
                print(
                    f"[SOLVER OK] "
                    f"run={ctx.run_id} "
                    f"problem={problem.id} "
                    f"solver={agent.config.llm_id}"
                )
