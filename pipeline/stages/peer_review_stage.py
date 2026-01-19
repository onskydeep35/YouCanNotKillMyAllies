import asyncio
from typing import List, Dict

from llm.agent import LLMAgent
from llm.models.dataclass.problem import Problem
from llm.models.pydantic.problem_solution import ProblemSolution
from pipeline.run_context import RunContext

from data.firestore_writer import FirestoreWriter
from data.firestore_writer import SOLUTIONS, PEER_REVIEWS

class PeerReviewStage:
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
        """
        Each solver reviews all other solvers' solutions.
        """

        # Only solvers participate
        solver_agents = [
            a for a in self.agents
            if ctx.final_roles.get(a.config.llm_id) == "Solver"
        ]

        if len(solver_agents) < 2:
            raise RuntimeError("Peer review requires at least 2 solvers")

        for problem in problems:
            solutions = await self._load_solutions(
                ctx.run_id, problem.id
            )

            if len(solutions) < 2:
                raise RuntimeError(
                    f"Not enough solutions for peer review: {problem.id}"
                )

            await self._run_peer_reviews_for_problem(
                ctx=ctx,
                problem=problem,
                solver_agents=solver_agents,
                solutions=solutions,
                timeout_sec=timeout_sec,
                log_interval_sec=log_interval_sec,
            )
