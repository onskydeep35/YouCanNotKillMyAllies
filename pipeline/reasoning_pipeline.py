import json
import asyncio
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

from llm.models.problem import Problem
from llm.models.agent_config import LLMAgentConfig
from llm.models.role_assessment import RoleAssessment
from llm.models.roles import LLMRolePreference

from llm.agent import LLMAgent
from llm.client import create_gemini_client
from llm.prompts import build_system_prompt


class ReasoningPipeline:
    """
    Coordinates multi-agent reasoning:
    - Stage 0: Role self-assessment (LLM)
    - Stage 0.5: Deterministic role assignment (code)
    - Stage 1: Solver execution (LLM)
    """

    def __init__(self, *, problems_path: str, problem_start : int, problem_count : int, output_dir: str = "data/temp"):
        self.problems_path = problems_path
        self.output_dir = Path(output_dir)

        self.problems: List[Problem] = []
        self.agents: List[LLMAgent] = []

        self.problem_start = problem_start
        self.problem_count = problem_count

        # Stage 0 outputs
        self.role_assessments: Dict[str, List[RoleAssessment]] = {}

        # Stage 0.5 outputs
        self.final_roles: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_problems(self) -> None:
        with open(self.problems_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.problems = [Problem.from_dict(p) for p in raw]
        self.problems = self.problems[self.problem_start : self.problem_start + self.problem_count]

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def create_llm_configs() -> List[LLMAgentConfig]:
        """
        Static factory for LLM agent configurations.
        """
        return [
            LLMAgentConfig(
                llm_id="gemini_1",
                model="gemini-3-flash-preview",
                temperature=0.4,
                top_p=0.9,
            ),
            LLMAgentConfig(
                llm_id="gemini_2",
                model="gemini-3-pro-preview",
                temperature=0.6,
                top_p=0.9,
            ),
        ]

    def create_agents(self) -> None:
        client = create_gemini_client()
        configs = self.create_llm_configs()

        self.agents = [
            LLMAgent(client=client, config=cfg)
            for cfg in configs
        ]

    # ------------------------------------------------------------------
    # Stage 0: Role self-assessment (LLM)
    # ------------------------------------------------------------------

    async def run_role_assessment_phase(
        self,
        *,
        timeout_sec: int = 300,
        log_interval_sec: int = 5,
        max_concurrency: int = 4,
    ) -> None:
        """
        Each agent self-assesses suitability for Solver/Judge roles.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def assess(agent: LLMAgent, problem: Problem):
            async with semaphore:
                return await agent.assess_roles_for_problem(
                    problem=problem,
                    timeout_sec=timeout_sec,
                    log_interval_sec=log_interval_sec,
                )

        for problem in self.problems:
            tasks = [assess(agent, problem) for agent in self.agents]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            assessments: List[RoleAssessment] = []
            for agent, result in zip(self.agents, results):
                if isinstance(result, Exception):
                    print(
                        f"[FAIL] role assessment agent={agent.config.llm_id} "
                        f"problem={problem.id} error={result}"
                    )
                    continue
                assessments.append(result)

            self.role_assessments[problem.id] = assessments

    # ------------------------------------------------------------------
    # Stage 0.5: Deterministic role assignment (NO LLMs)
    # ------------------------------------------------------------------

    def assign_roles_deterministically(self) -> None:
        """
        Deterministically assigns roles based on Solver confidence.
        """
        if not self.role_assessments:
            raise RuntimeError("Role assessments not available")

        # Use first problem as reference (roles are global for now)
        assessments = next(iter(self.role_assessments.values()))

        sorted_agents = sorted(
            assessments,
            key=lambda a: (
                a.role_scores.get("Solver", 0.0),
                a.llm_id,  # tie-breaker
            ),
            reverse=True,
        )

        num_solvers = (len(sorted_agents) + 1) // 2

        for idx, assessment in enumerate(sorted_agents):
            role = "Solver" if idx < num_solvers else "Judge"
            self.final_roles[assessment.llm_id] = role

        print("\n[FINAL ROLE ASSIGNMENT]")
        for llm_id, role in self.final_roles.items():
            print(f"  {llm_id} -> {role}")

    # ------------------------------------------------------------------
    # Stage 1: Solver phase (LLM)
    # ------------------------------------------------------------------

    async def run_solver_phase(
        self,
        *,
        timeout_sec: int = 2000,
        log_interval_sec: int = 5,
        max_concurrency: int = 4,
    ) -> None:
        """
        Runs solver agents only.
        """
        solver_role = LLMRolePreference(
            role_preferences=["Solver"],
            confidence_by_role={"Solver": 1.0},
            reasoning="Solve the problem using precise logical reasoning.",
        )

        system_prompt = build_system_prompt(solver_role)
        semaphore = asyncio.Semaphore(max_concurrency)

        solver_agents = [
            agent for agent in self.agents
            if self.final_roles.get(agent.config.llm_id) == "Solver"
        ]

        async def solve(agent: LLMAgent, problem: Problem):
            async with semaphore:
                return await agent.solve_problem(
                    problem=problem,
                    system_prompt=system_prompt,
                    timeout_sec=timeout_sec,
                    log_interval_sec=log_interval_sec,
                )

        tasks = []
        for agent in solver_agents:
            for problem in self.problems:
                tasks.append((agent, problem))

        results = await asyncio.gather(
            *[solve(a, p) for a, p in tasks],
            return_exceptions=True,
        )

        self._persist_results(tasks, results)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_results(
        self,
        tasks: List[tuple],
        results: List,
    ) -> None:
        for idx, ((agent, problem), result) in enumerate(zip(tasks, results)):
            if isinstance(result, Exception):
                print(
                    f"[FAIL] agent={agent.config.llm_id} "
                    f"problem={problem.id} error={result}"
                )
                continue

            payload = {
                "agent_id": agent.config.llm_id,
                "assigned_role": self.final_roles.get(agent.config.llm_id),
                "problem_id": problem.id,
                "problem": problem.__dict__,
                "solver_output": result.model_dump(),
            }

            out_path = self.output_dir / f"solution_{idx}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            print(f"[OK] {out_path.name} written")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        load_dotenv()
        self.load_problems()
        self.ensure_output_dir()
        self.create_agents()

        # Stage 0
        await self.run_role_assessment_phase()

        # Stage 0.5
        self.assign_roles_deterministically()

        # Stage 1
        #await self.run_solver_phase()
