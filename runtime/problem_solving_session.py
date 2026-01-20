import asyncio
import json
from pathlib import Path
from typing import List

from llm.agents.agent import LLMAgent
from schemas.pydantic.problem import Problem
from schemas.pydantic.role_assessment import RoleAssessment
from schemas.pydantic.problem_solution_review import ProblemSolutionReview
from schemas.pydantic.refined_problem_solution import RefinedProblemSolution

from data.persistence.firestore_writer import (
    FirestoreWriter,
    SOLUTIONS,
    SOLUTION_REVIEWS,
    REFINED_SOLUTIONS,
    ROLE_ASSESSMENTS,
)

from runtime.contexts.solver_agent_context import SolverAgentContext


class ProblemSolvingSession:
    """
    Executes a full debate for a single problem.
    """

    def __init__(
        self,
        *,
        run_id: str,
        problem: Problem,
        agents: List[LLMAgent],
        writer: FirestoreWriter,
        output_dir: Path,
        max_concurrency: int = 4,
    ):
        self.run_id = run_id
        self.problem = problem
        self.agents = agents
        self.writer = writer
        self.output_dir = output_dir
        self.semaphore = asyncio.Semaphore(max_concurrency)

        self.solver_contexts: List[SolverAgentContext] = []
        self.judge_agents: List[LLMAgent] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        *,
        timeout_sec: int,
        log_interval_sec: int,
    ) -> None:

        print(f"[SESSION START] problem={self.problem.problem_id}")

        await self._assign_roles(
            timeout_sec=timeout_sec,
            log_interval_sec=log_interval_sec,
        )

        await self._run_solvers(
            timeout_sec=timeout_sec,
            log_interval_sec=log_interval_sec,
        )

        await self._run_peer_reviews(
            timeout_sec=timeout_sec,
            log_interval_sec=log_interval_sec,
        )

        await self._run_refinements(
            timeout_sec=timeout_sec,
            log_interval_sec=log_interval_sec,
        )

        print(f"[SESSION END] problem={self.problem.problem_id}")

    # -------------------------
    # Stage 0: Role assignment
    # -------------------------
    async def _assign_roles(
        self,
        *,
        timeout_sec: int,
        log_interval_sec: int,
    ) -> None:

        print("[ROLE ASSESSMENT START]")

        assessment_dir = self.output_dir / "role_assessments"
        assessment_dir.mkdir(parents=True, exist_ok=True)

        contexts = [
            SolverAgentContext(
                agent=a,
                problem=self.problem,
                run_id=self.run_id,
                output_dir=self.output_dir,
            )
            for a in self.agents
        ]

        async def assess(ctx: SolverAgentContext) -> RoleAssessment:
            async with self.semaphore:
                assessment = await ctx.assess_role(
                    timeout_sec=timeout_sec,
                    log_interval_sec=log_interval_sec,
                )

                document = {
                    "llm_id": assessment.llm_id,
                    "assessment_id": assessment.assessment_id,
                    "problem_id": assessment.problem_id,
                    "run_id": assessment.run_id,
                    **assessment.model_dump(),
                }

                await self.writer.write(
                    collection=ROLE_ASSESSMENTS,
                    document=document,
                    document_id=assessment.assessment_id,
                )

                file_path = (
                    assessment_dir /
                    f"{assessment.llm_id}_{assessment.problem_id}.json"
                )

                file_path.write_text(
                    json.dumps(document, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                return assessment

        results = await asyncio.gather(*[assess(c) for c in contexts])

        if len(results) < 3:
            raise RuntimeError("At least 3 agents required")

        def judge_preference(a: RoleAssessment) -> float:
            return a.judge_score - a.solver_score

        results.sort(key=judge_preference, reverse=True)

        judge_id = results[0].llm_id
        solver_ids = {a.llm_id for a in results[1:]}

        print("[ROLE ASSIGNMENT]")
        print(f"  Judge : {judge_id}")
        for sid in solver_ids:
            print(f"  Solver: {sid}")

        for ctx in contexts:
            if ctx.solver_id == judge_id:
                self.judge_agents.append(ctx.agent)
            elif ctx.solver_id in solver_ids:
                self.solver_contexts.append(ctx)

    # -------------------------
    # Stage 1: Solving
    # -------------------------
    async def _run_solvers(
        self,
        *,
        timeout_sec: int,
        log_interval_sec: int,
    ) -> None:

        async def solve(ctx: SolverAgentContext) -> None:
            async with self.semaphore:
                solution = await ctx.solve(
                    timeout_sec=timeout_sec,
                    log_interval_sec=log_interval_sec,
                )

                document = {
                    "run_id": solution.run_id,
                    "solution_id": solution.solution_id,
                    "problem_id": solution.problem_id,
                    "solver_llm_model_id": solution.solver_llm_model_id,
                    "time_elapsed_sec": solution.time_elapsed_sec,
                    **solution.model_dump(),
                }

                print("[SOLUTION DOCUMENT]", document)

                await self.writer.write(
                    collection=SOLUTIONS,
                    document=document,
                    document_id=solution.solution_id
                )

                solutions_dir = self.output_dir / "solutions"
                solutions_dir.mkdir(parents=True, exist_ok=True)

                file_path = (
                        solutions_dir /
                        f"{ctx.solver_id}_{self.problem.problem_id}.json"
                )

                file_path.write_text(
                    json.dumps(document, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        await asyncio.gather(
            *[solve(ctx) for ctx in self.solver_contexts]
        )

    # -------------------------
    # Stage 2: Peer review
    # -------------------------
    async def _run_peer_reviews(
        self,
        *,
        timeout_sec: int,
        log_interval_sec: int,
    ) -> None:

        if len(self.solver_contexts) < 2:
            print("[PEER REVIEW SKIPPED] Not enough solvers")
            return

        print("[PEER REVIEW START]")

        review_dir = self.output_dir / "reviews"
        review_dir.mkdir(parents=True, exist_ok=True)

        async def review(
            reviewer: SolverAgentContext,
            reviewee: SolverAgentContext,
        ) -> None:
            async with self.semaphore:
                review: ProblemSolutionReview = await reviewer.generate_review(
                    solution=reviewee.solution,
                    timeout_sec=timeout_sec,
                    log_interval_sec=log_interval_sec,
                )

                reviewee.receive_review(review=review)

                document = {
                    "review_id": review.review_id,
                    "run_id": review.run_id,
                    "problem_id": review.problem_id,
                    "reviewer_id": review.reviewer_id,
                    "reviewee_id": review.reviewee_id,
                    **review.model_dump(),
                }

                await self.writer.write(
                    collection=SOLUTION_REVIEWS,
                    document=document,
                    document_id=review.review_id
                )

                solutions_dir = self.output_dir / "reviews"
                solutions_dir.mkdir(parents=True, exist_ok=True)

                file_path = (
                        solutions_dir /
                        f"{review.reviewer_id}_{review.reviewee_id}_{self.problem.problem_id}.json"
                )

                file_path.write_text(
                    json.dumps(document, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        await asyncio.gather(
            *[
                review(r, e)
                for r in self.solver_contexts
                for e in self.solver_contexts
                if r.solver_id != e.solver_id
            ]
        )

        print("[PEER REVIEW COMPLETE]")

    async def _run_refinements(
            self,
            *,
            timeout_sec: int,
            log_interval_sec: int,
    ) -> None:

        print("[REFINEMENT START]")

        refined_dir = self.output_dir / "refined_solutions"
        refined_dir.mkdir(parents=True, exist_ok=True)

        async def refine(ctx: SolverAgentContext) -> None:
            async with self.semaphore:
                if not ctx.peer_reviews:
                    print(
                        f"[REFINEMENT SKIPPED] solver={ctx.solver_id} (no peer reviews)"
                    )
                    return

                refined: RefinedProblemSolution = await ctx.refine_solution(
                    timeout_sec=timeout_sec,
                    log_interval_sec=log_interval_sec,
                )

                document = {
                    "run_id": refined.run_id,
                    "problem_id": refined.problem_id,
                    "solver_llm_model_id": refined.solver_llm_model_id,
                    "parent_solution_id": refined.parent_solution_id,
                    "refined_solution_id": refined.refined_solution_id,
                    "review_ids": refined.review_ids,
                    "time_elapsed_sec": refined.time_elapsed_sec,
                    **refined.model_dump(),
                }

                print("[REFINED SOLUTION DOCUMENT]", document)

                await self.writer.write(
                    collection=REFINED_SOLUTIONS,
                    document=document,
                    document_id=refined.refined_solution_id
                )

                file_path = (
                        refined_dir /
                        f"{ctx.solver_id}_{ctx.problem.problem_id}.json"
                )

                file_path.write_text(
                    json.dumps(document, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        await asyncio.gather(
            *[refine(ctx) for ctx in self.solver_contexts]
        )

        print("[REFINEMENT COMPLETE]")