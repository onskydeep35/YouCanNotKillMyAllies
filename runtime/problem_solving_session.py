import asyncio
from pathlib import Path
from datetime import datetime

from llm.agents.agent import LLMAgent
from schemas.pydantic.output.role_assessment import RoleAssessment
from schemas.pydantic.output.final_judgement import FinalJudgement
from schemas.utilities import *

from data.persistence.firestore_writer import (
    FirestoreWriter,
    SOLUTIONS,
    SOLUTION_REVIEWS,
    REFINED_SOLUTIONS,
    ROLE_ASSESSMENTS,
    FINAL_JUDGEMENTS,
    RUNS,
)

from runtime.contexts.solver_agent_context import SolverAgentContext
from runtime.contexts.judge_agent_context import JudgeAgentContext
from llm.prompts.prompts import *


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
        self.judge_context: Optional[JudgeAgentContext] = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        *,
        timeout_sec: int,
        log_interval_sec: int,
    ) -> None:

        print(f"[SESSION START] problem={self.problem.problem_id}")

        await self._persist_run()

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

        await self._run_final_judgement(
            timeout_sec=timeout_sec,
            log_interval_sec=log_interval_sec,
        )

        print(f"[SESSION END] problem={self.problem.problem_id}")

    async def _persist_run(self):
        document = {
            "run_id": self.run_id,
            "timestamp": datetime.now(),
        }

        await self.writer.write(
            collection=RUNS, document=document, document_id=self.run_id
        )

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

        async def assess(ctx: SolverAgentContext) -> Optional[RoleAssessment]:
            """Assess agent, return None if agent fails"""
            async with self.semaphore:
                try:
                    assessment = await ctx.assess_role(
                        timeout_sec=timeout_sec,
                        log_interval_sec=log_interval_sec,
                    )

                    document = PydanticSchemaUtils.build_full_document(assessment)

                    await self.writer.write(
                        collection=ROLE_ASSESSMENTS,
                        document=document,
                        document_id=assessment.assessment_id,
                    )

                    file_path = (
                        assessment_dir / f"{assessment.llm_id}_{assessment.problem_id}.json"
                    )

                    file_path.write_text(
                        json.dumps(document, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

                    return assessment
                
                except Exception as e:
                    print(f"[AGENT EXCLUDED] {ctx.solver_id} failed: {type(e).__name__}")
                    return None

        results = await asyncio.gather(*[assess(c) for c in contexts])

        # Filter out failed agents
        available_agents = [r for r in results if r is not None]

        if len(available_agents) < 4:
            raise RuntimeError(
                f"Insufficient working agents: {len(available_agents)}/{len(results)} succeeded (need 4)"
            )

        print(f"[AVAILABLE] {len(available_agents)}/{len(results)} agents ready")

        # Select top 4
        def overall_capability(a: RoleAssessment) -> float:
            return max(a.judge_score, a.solver_score)

        available_agents.sort(key=overall_capability, reverse=True)
        top_4 = available_agents[:4]

        print(f"[SELECTED TOP 4]")
        for idx, a in enumerate(top_4, start=1):
            print(f"  #{idx}: {a.llm_id} (score: {overall_capability(a):.2f})")

        # Assign roles within top 4
        def judge_preference(a: RoleAssessment) -> float:
            """
            Weight judge preference by confidence.
            Strong preferences (0.9 vs 0.1) beat weak ones (0.51 vs 0.49).
            """
            score_diff = a.judge_score - a.solver_score
            confidence = abs(score_diff)
            return score_diff * (1 + confidence)

        top_4.sort(key=judge_preference, reverse=True)

        judge_id = top_4[0].llm_id
        solver_ids = [a.llm_id for a in top_4[1:4]]

        print("[ROLE ASSIGNMENT]")
        print(f"  Judge : {judge_id}")
        for i, sid in enumerate(solver_ids, start=1):
            print(f"  Solver {i}: {sid}")

        solver_contexts: List[SolverAgentContext] = []

        for ctx in contexts:
            if ctx.solver_id in solver_ids:
                solver_contexts.append(ctx)
            elif ctx.solver_id == judge_id:
                self.judge_context = JudgeAgentContext(
                    agent=ctx.agent,
                    problem=self.problem,
                    run_id=self.run_id,
                )

        # preserve solver ordering as Solver 1/2/3
        self.solver_contexts = solver_contexts

        # now inject solver contexts into judge
        self.judge_context.solver_contexts = self.solver_contexts

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

                document = PydanticSchemaUtils.build_full_document(solution)

                await self.writer.write(
                    collection=SOLUTIONS,
                    document=document,
                    document_id=solution.solution_id,
                )

                solutions_dir = self.output_dir / "solutions"
                solutions_dir.mkdir(parents=True, exist_ok=True)

                file_path = (
                    solutions_dir / f"{ctx.solver_id}_{self.problem.problem_id}.json"
                )

                file_path.write_text(
                    json.dumps(document, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        await asyncio.gather(*[solve(ctx) for ctx in self.solver_contexts])

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

                document = PydanticSchemaUtils.build_full_document(review)

                await self.writer.write(
                    collection=SOLUTION_REVIEWS,
                    document=document,
                    document_id=review.review_id,
                )

                solutions_dir = self.output_dir / "reviews"
                solutions_dir.mkdir(parents=True, exist_ok=True)

                file_path = (
                    solutions_dir
                    / f"{review.reviewer_id}_{review.reviewee_id}_{self.problem.problem_id}.json"
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

                refined_solution: RefinedProblemSolution = await ctx.refine_solution(
                    timeout_sec=timeout_sec,
                    log_interval_sec=log_interval_sec,
                )

                document = PydanticSchemaUtils.build_full_document(refined_solution)

                print("[REFINED SOLUTION DOCUMENT]", document)

                await self.writer.write(
                    collection=REFINED_SOLUTIONS,
                    document=document,
                    document_id=refined_solution.refined_solution_id,
                )

                file_path = (
                    refined_dir / f"{ctx.solver_id}_{ctx.problem.problem_id}.json"
                )

                file_path.write_text(
                    json.dumps(document, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        await asyncio.gather(*[refine(ctx) for ctx in self.solver_contexts])

        print("[REFINEMENT COMPLETE]")

    async def _run_final_judgement(
        self,
        *,
        timeout_sec: int,
        log_interval_sec: int,
    ) -> None:

        print("[FINAL JUDGMENT START]")

        if not self.judge_context:
            raise RuntimeError("JudgeAgentContext not initialized")

        judgement: FinalJudgement = await self.judge_context.generate_judgement(
            solver_agent_contexts=self.solver_contexts,
            timeout_sec=timeout_sec,
            log_interval_sec=log_interval_sec,
        )

        document = PydanticSchemaUtils.build_full_document(judgement)

        await self.writer.write(
            collection=FINAL_JUDGEMENTS,
            document=judgement.model_dump(),
            document_id=judgement.judgement_id,
        )

        judgement_dir = self.output_dir / "final_judgements"
        judgement_dir.mkdir(parents=True, exist_ok=True)

        file_path = (
            judgement_dir
            / f"{self.judge_context.judge_id}_{self.problem.problem_id}.json"
        )

        file_path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        winner_index = int(judgement.winner_solver.split()[-1]) - 1
        winner_ctx = self.solver_contexts[winner_index]

        print("[FINAL JUDGMENT COMPLETE]")
        print("WINNER:", judgement.winner_solver)
        print("FINAL ANSWER:")
        print(winner_ctx.refined_solution.refined_answer)