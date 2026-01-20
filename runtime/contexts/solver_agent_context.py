from pathlib import Path
import uuid
from llm.agents.agent import LLMAgent
from schemas.pydantic.refined_problem_solution import *
from llm.prompts.prompts import *


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
        self.refined_solution: Optional[RefinedProblemSolution] = None

        self.output_dir.mkdir(parents=True, exist_ok=True)


    @property
    def solver_id(self) -> str:
        return self.agent.config.llm_id


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

    async def refine_solution(
            self,
            *,
            timeout_sec: int,
            log_interval_sec: int,
    ) -> RefinedProblemSolution:
        """
        Stage 3: Refine the previously generated solution using peer reviews.
        """

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
            review.review_id
            for review in self.peer_reviews
            if review.review_id is not None
        ]

        self.refined_solution = refined_solution
        return refined_solution
