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

    # -------------------------
    # Setup
    # -------------------------
    def load_problems(self) -> None:
        raw = json.loads(Path(self.problems_path).read_text())
        self.problems = [
            Problem.from_dict(p)
            for p in raw
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
            LLMAgentConfig(
                llm_id="pro_conservative",
                model="gemini-3-pro-preview",
                temperature=0.2,
                top_p=0.85,
            ),

            LLMAgentConfig(
                llm_id="pro_exploratory",
                model="gemini-3-pro-preview",
                temperature=0.5,
                top_p=0.9,
            ),

            LLMAgentConfig(
                llm_id="flash_low_var",
                model="gemini-3-flash-preview",
                temperature=0.3,
                top_p=0.85,
            ),

            LLMAgentConfig(
                llm_id="flash_high_var",
                model="gemini-3-flash-preview",
                temperature=0.7,
                top_p=0.95,
            ),
        ]

    # -------------------------
    # Main pipeline
    # -------------------------
    async def run(self) -> None:
        load_dotenv()

        self.load_problems()
        self.create_agents()

        # Firestore setup (once per run)
        db = get_firestore_client()
        writer = FirestoreWriter(db)

        role_stage = RoleDeterminationStage(self.agents)
        solver_stage = SolverStage(
            self.agents,
            writer=writer,
        )

        for idx, problem in enumerate(self.problems, start=1):
            print(
                f"\n========== "
                f"PROCESSING PROBLEM {idx}/{len(self.problems)} "
                f"(id={problem.id}) "
                f"==========\n"
            )

            # New context per problem
            ctx = RunContext(run_id=f"{idx}-{problem.id}")

            # Stage 0: role determination
            await role_stage.run(
                ctx=ctx,
                problem=problem,
                timeout_sec=300,
                log_interval_sec=5,
            )

            # Stage 1: independent solvers
            await solver_stage.run(
                ctx=ctx,
                problem=problem,
                timeout_sec=2000,
                log_interval_sec=5,
            )

            print(
                f"\n========== "
                f"FINISHED PROBLEM {idx}/{len(self.problems)} "
                f"(id={problem.id}) "
                f"==========\n"
            )