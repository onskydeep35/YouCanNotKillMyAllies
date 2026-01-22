import uuid

from llm.agents.agent import LLMAgent
from schemas.pydantic.output.final_judgement import FinalJudgement

from runtime.contexts.solver_agent_context import SolverAgentContext
from llm.prompts.prompts import *

class JudgeAgentContext:
    """
    Holds judge-related state and behavior for Stage 4: Final Judgment.

    The Judge evaluates:
    - The original problem
    - All solver original solutions
    - All peer reviews
    - All refined solutions

    And selects exactly one winning solver.
    """

    def __init__(
        self,
        *,
        agent: LLMAgent,
        problem: Problem,
        run_id: str,
    ):
        self.agent = agent
        self.problem = problem
        self.run_id = run_id

        self.judgment: Optional[FinalJudgement] = None

    @property
    def judge_id(self) -> str:
        return self.agent.config.llm_id

    @staticmethod
    def build_final_judgement_input(
            *,
            problem: Problem,
            solver_agent_contexts: List[SolverAgentContext],
    ) -> FinalJudgementInput:
        """
        Constructs the FinalJudgementInput object from solver agent contexts.
        """

        solver_contexts: List[SolverContexts] = []

        for ctx in solver_agent_contexts:
            solver_contexts.append(
                SolverContexts(
                    solution=ctx.solution,
                    received_reviews=ctx.peer_reviews,
                    refined_solution=ctx.refined_solution,
                )
            )

        return FinalJudgementInput(
            problem=problem,
            solver_contexts=solver_contexts,
        )

    async def generate_judgement(
        self,
        *,
        solver_agent_contexts: List[SolverAgentContext],
        timeout_sec: int,
        log_interval_sec: int,
    ) -> FinalJudgement:
        """
        Executes Stage 4: Final Judgement.

        Returns a fully populated FinalJudgement object.
        """

        # --- Safety checks ---
        for idx, ctx in enumerate(solver_agent_contexts, start=1):
            if ctx.solution is None:
                raise RuntimeError(
                    f"Solver {idx} has no original solution"
                )
            if ctx.refined_solution is None:
                raise RuntimeError(
                    f"Solver {idx} has no refined solution"
                )

        system_prompt = FINAL_JUDGMENT_SYSTEM_PROMPT

        final_input = self.build_final_judgement_input(
            problem=self.problem,
            solver_agent_contexts=solver_agent_contexts,
        )

        user_prompt = build_final_judgement_user_prompt(
            final_input=final_input,
        )

        judgement = await self.agent.run_structured_call(
            problem=self.problem,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=FinalJudgement,
            method_type="final_judgment",
            timeout_sec=timeout_sec,
            log_interval_sec=log_interval_sec,
        )

        judgement.prompt_system = system_prompt
        judgement.prompt_user = user_prompt
        judgement.llm_id = self.judge_id
        judgement.run_id = self.run_id
        judgement.problem_id = self.problem.problem_id
        judgement.judgement_id = uuid.uuid4().hex
        judgement.time_elapsed_sec = judgement.time_elapsed_sec

        self.judgment = judgement
        return judgement

