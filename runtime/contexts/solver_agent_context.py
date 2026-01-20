from pathlib import Path
import uuid
from typing import Optional

from llm.agents.agent import LLMAgent
from schemas.pydantic.problem import Problem
from schemas.pydantic.role_assessment import RoleAssessment
from schemas.pydantic.problem_solution_review import ProblemSolutionReview
from schemas.pydantic.refined_problem_solution import RefinedProblemSolution
from schemas.pydantic.problem_solution import ProblemSolution

from llm.prompts.prompts import (
    ROLE_DETERMINATION_SYSTEM_PROMPT,
    build_role_determination_user_prompt,
    build_solver_system_prompt,
    build_solver_user_prompt,
    PEER_REVIEW_SYSTEM_PROMPT,
    build_peer_review_user_prompt,
    REFINE_SOLUTION_SYSTEM_PROMPT,
    build_solution_refinement_user_prompt,
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

        self.role_assessment: Optional[RoleAssessment] = None
        self.solution: Optional[ProblemSolution] = None
        self.peer_reviews: list[ProblemSolutionReview] = []
        self.refined_solution: Optional[RefinedProblemSolution] = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def solver_id(self) -> str:
        return self.agent.config.llm_id

    # -------------------------
    # Stage 0: Role assessment
    # -------------------------
    async def assess_role(
        self,
        *,
        timeout_sec: int,
        log_interval_sec: int,
    ) -> RoleAssessment:
        """
        Performs LLM role self-assessment for this agent.
        """

        assessment = await self.agent.run_structured_call(
            problem=self.problem,
            system_prompt=ROLE_DETERMINATION_SYSTEM_PROMPT,
            user_prompt=build_role_determination_user_prompt(self.problem),
            output_model=RoleAssessment,
            method_type="role_determination",
            timeout_sec=timeout_sec,
            log_interval_sec=log_interval_sec,
        )

        assessment.llm_id = self.solver_id
        assessment.run_id = self.run_id
        assessment.assessment_id = uuid.uuid4().hex
        assessment.problem_id = self.problem.problem_id

        self.role_assessment = assessment
        return assessment

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
        solution.solver_llm_model_id = self.solver_id
        solution.solution_id = uuid.uuid4().hex
        solution.problem_id = self.problem.problem_id

        self.solution = solution
        return solution

    # -------------------------
    # Stage 2: Peer review
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
        review.solution_id = solution.solution_id
        review.reviewee_id = solution.solver_llm_model_id

        return review

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
            f"confidence={review.confidence:.2f}"
        )

    # -------------------------
    # Stage 3: Refinement
    # -------------------------
    async def refine_solution(
        self,
        *,
        timeout_sec: int,
        log_interval_sec: int,
    ) -> RefinedProblemSolution:

        refined_solution = await self.agent.run_structured_call(
            problem=self.problem,
            system_prompt=REFINE_SOLUTION_SYSTEM_PROMPT,
            user_prompt=build_solution_refinement_user_prompt(
                problem=self.problem,
                initial_solution=self.solution,
                reviews=self.peer_reviews,
            ),
            output_model=RefinedProblemSolution,
            method_type="solution_refinement",
            timeout_sec=timeout_sec,
            log_interval_sec=log_interval_sec,
        )

        refined_solution.run_id = self.run_id
        refined_solution.solver_llm_model_id = self.solver_id
        refined_solution.parent_solution_id = self.solution.solution_id
        refined_solution.refined_solution_id = uuid.uuid4().hex
        refined_solution.problem_id = self.problem.problem_id
        refined_solution.review_ids = [
            r.review_id for r in self.peer_reviews if r.review_id
        ]

        self.refined_solution = refined_solution
        return refined_solution
