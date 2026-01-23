import json
from pathlib import Path
from typing import List
import uuid
import asyncio

from dotenv import load_dotenv

from llm.agents.agent import LLMAgent
from llm.agents.agent_factory import AgentFactory
from schemas.pydantic.input.problem import Problem
from schemas.dataclass.agent_config import LLMAgentConfig
from data.persistence.firestore_client import get_firestore_client
from data.persistence.firestore_manager import FirestoreManager

from runtime.problem_solving_session import ProblemSolvingSession


class ProblemSolvingApp:
    """
    Top-level execution session.
    Loads problems, creates agents once,
    and runs a DebateSession per problem.
    """

    def __init__(
        self,
        *,
        problems_path: Path,
        agent_configs: List[LLMAgentConfig],
        problems_skip: int = 0,
        problems_take: int | None = None,
        output_dir: Path = Path("data/output"),
    ):
        self.run_id = uuid.uuid4().hex

        self.problems_path = problems_path
        self.problems_skip = problems_skip
        self.problems_take = problems_take

        self.agent_configs = agent_configs
        self.output_dir = output_dir

        self.problems: List[Problem] = []
        self.agents: List[LLMAgent] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Setup
    # -------------------------
    def load_problems(self) -> None:
        raw = json.loads(self.problems_path.read_text())

        problems = [Problem.model_validate(p) for p in raw]

        if self.problems_skip:
            problems = problems[self.problems_skip :]

        if self.problems_take is not None:
            problems = problems[: self.problems_take]

        self.problems = problems

    def create_agents(self) -> None:
        self.agents = AgentFactory.create_agents(self.agent_configs)

    # -------------------------
    # Main entry point
    # -------------------------
    async def run(
        self,
        *,
        timeout_sec: int,
        log_interval_sec: int,
        max_concurrent_sessions: int = 2,
    ) -> None:
        load_dotenv()

        self.load_problems()
        self.create_agents()

        # Firestore lifecycle: once per run
        db = get_firestore_client()
        firestore_manager = FirestoreManager(db)

        semaphore = asyncio.Semaphore(max_concurrent_sessions)

        print(
            f"\n[RUN START] "
            f"run_id={self.run_id} "
            f"problems={len(self.problems)} "
            f"agents={len(self.agents)} "
            f"max_concurrency={max_concurrent_sessions}\n"
        )

        async def run_single_problem(idx: int, problem: Problem):
            async with semaphore:
                print(
                    f"\n========== "
                    f"PROBLEM {idx}/{len(self.problems)} "
                    f"id={problem.problem_id} "
                    f"==========\n"
                )

                session = ProblemSolvingSession(
                    run_id=self.run_id,
                    problem=problem,
                    agents=self.agents,
                    firestore_manager=firestore_manager,
                    output_dir=self.output_dir,
                )

                await session.run(
                    timeout_sec=timeout_sec,
                    log_interval_sec=log_interval_sec,
                )

                print(
                    f"\n========== "
                    f"FINISHED PROBLEM {idx}/{len(self.problems)} "
                    f"id={problem.problem_id} "
                    f"==========\n"
                )

        tasks = [
            asyncio.create_task(run_single_problem(idx, problem))
            for idx, problem in enumerate(self.problems, start=1)
        ]

        await asyncio.gather(*tasks)

        print(f"\n[RUN END] run_id={self.run_id}\n")
