import json
from pathlib import Path
from typing import Optional
import uuid
from llm.agents.agent import LLMAgent
from schemas.dataclass.problem import Problem
from schemas.pydantic.problem_solution import ProblemSolution
from schemas.pydantic.problem_solution_review import ProblemSolutionReview
from llm.prompts.prompts import (
    build_solver_system_prompt,
    build_solver_user_prompt,
    PEER_REVIEW_SYSTEM_PROMPT,
    build_peer_review_user_prompt,
)


class SolverAgentContext:
    """
    Holds all solver-related state and behavior
    for a single problem.
    """

    def __init__(
        self,
        *,
        agent: LLMAgent,
        problem: Problem,
        run_id: str,
        output_dir: Path,
    ):
        self.agent = agent
        self.problem = problem
        self.run_id = run_id
        self.output_dir = output_dir

        self.solution: Optional[ProblemSolution] = None
        self.peer_reviews: list[ProblemSolutionReview] = []
        self.refined_solution: Optional[ProblemSolution] = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Identity
    # -------------------------
    @property
    def solver_id(self) -> str:
        return self.agent.config.llm_id

    # -------------------------
    # Stage 1: Solve
    # -------------------------
    async def solve(
        self,
        *,
        timeout_sec: int,
        log_interval_sec: int,
    ) -> ProblemSolution:

        solution = await self.agent.run_structured_call(
            problem=self.problem,
            system_prompt=build_solver_system_prompt(
                category=self.problem.category
            ),
            user_prompt=build_solver_user_prompt(self.problem),
            output_model=ProblemSolution,
            method_type="solver",
            timeout_sec=timeout_sec,
            log_interval_sec=log_interval_sec,
        )

        solution.run_id = self.run_id
        solution.solver_llm_model_id = self.agent.config.llm_id
        solution.solution_id = uuid.uuid4().hex
        solution.problem_id = self.problem.problem_id

        self.solution = solution
        return solution

    # -------------------------
    # Stage 2: Generate review
    # -------------------------
    async def generate_review(
        self,
        *,
        solution: ProblemSolution,
        timeout_sec: int,
        log_interval_sec: int,
    ) -> ProblemSolutionReview:

        review = await self.agent.run_structured_call(
            problem=self.problem,
            system_prompt=PEER_REVIEW_SYSTEM_PROMPT,
            user_prompt=build_peer_review_user_prompt(
                problem=self.problem,
                solution=solution,
            ),
            output_model=ProblemSolutionReview,
            method_type="peer_review",
            timeout_sec=timeout_sec,
            log_interval_sec=log_interval_sec,
        )

        review.review_id = uuid.uuid4().hex
        review.run_id = self.run_id
        review.problem_id = self.problem.problem_id
        review.reviewer_id = self.solver_id
        review.reviewee_id = solution.solver_llm_model_id
        return review

    # -------------------------
    # Receive review
    # -------------------------
    def receive_review(
        self,
        *,
        review: ProblemSolutionReview,
    ) -> None:

        if review.reviewee_id != self.solver_id:
            raise ValueError(
                f"Review intended for '{review.reviewee_id}' "
                f"received by solver '{self.solver_id}'"
            )

        self.peer_reviews.append(review)

        print(
            f"[REVIEW RECEIVED] "
            f"solver={self.solver_id} "
            f"from={review.reviewer_id} "
            f"assessment={review.overall_assessment} "
            f"confidence={review.confidence_score:.2f}"
        )

    def solution_file_path(self) -> Path:
        return (
            self.output_dir /
            f"{self.run_id}_{self.solver_id}_{self.problem.problem_id}.json"
        )

    def review_file_path(self, *, reviewee_id: str) -> Path:
        review_dir = self.output_dir / "reviews"
        review_dir.mkdir(parents=True, exist_ok=True)

        return (
            review_dir /
            f"{self.run_id}_{self.solver_id}_reviews_{reviewee_id}.json"
        )
