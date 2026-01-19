import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from llm.agent import LLMAgent
from llm.client import create_gemini_client
from llm.models.dataclass.problem import Problem
from llm.models.dataclass.agent_config import LLMAgentConfig

from pipeline.run_context import RunContext
from pipeline.stages.role_determination_stage import RoleDeterminationStage
from pipeline.stages.problem_solver_stage import SolverStage

from data.firestore_client import get_firestore_client
from data.firestore_writer import FirestoreWriter


class DebatingPipeline:
    def __init__(
        self,
        *,
        problems_path: str,
        problems_skip: int,
        problems_take: int,
    ):
        self.problems_path = problems_path
        self.problems_skip = problems_skip
        self.problems_take = problems_take

        self.problems: List[Problem] = []
        self.agents: List[LLMAgent] = []

    def load_problems(self) -> None:
        raw = json.loads(Path(self.problems_path).read_text())
        self.problems = [
            Problem.from_dict(p) for p in raw
        ][self.problems_skip : self.problems_skip + self.problems_take]

    def create_agents(self) -> None:
        client = create_gemini_client()
        configs = self.create_llm_configs()
        self.agents = [
            LLMAgent(client=client, config=cfg)
            for cfg in configs
        ]

    @staticmethod
    def create_llm_configs() -> List[LLMAgentConfig]:
        return [
            LLMAgentConfig("gemini-3-flash", "gemini-2.5-flash-preview", 0.4, 0.9),
            LLMAgentConfig("gemini-3-pro", "gemini-3-pro-preview", 0.6, 0.9),
            LLMAgentConfig("gemini-2.0-flash", "gemini-2.0-flash", 0.5, 0.9),
            LLMAgentConfig("gemini-2.5-flash-lite", "gemini-2.5-flash-lite", 0.7, 0.9),
        ]

    async def run(self) -> None:
        load_dotenv()

        self.load_problems()
        self.create_agents()

        ctx = RunContext()

        # Firestore setup (once)
        db = get_firestore_client()
        writer = FirestoreWriter(db)

        # Stage 0 + 0.5
        ctx = RunContext()

        await RoleDeterminationStage(self.agents).run(
            ctx=ctx,
            problems=self.problems,
            timeout_sec=300,
            log_interval_sec=5,
        )

        await SolverStage(
            self.agents,
            writer=writer,
        ).run(
            ctx=ctx,
            problems=self.problems,
            timeout_sec=2000,
            log_interval_sec=5,
        )
