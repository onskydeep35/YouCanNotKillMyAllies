from schemas.pydantic.input.problem import *
from schemas.pydantic.input.peer_review_input import PeerReviewInput
from schemas.pydantic.input.solution_refinement_input import SolutionRefinementInput
from schemas.pydantic.input.final_judgement_input import *
from schemas.pydantic.output.problem_solution import *
from schemas.pydantic.output.problem_solution_review import *
from schemas.utilities.pydantic_schema_utils import PydanticSchemaUtils

import json

DEFAULT_SOLVER_POLICY = """
    You are a general-purpose problem solver.
    
    Focus on:
    - Correctness
    - Clear reasoning
    - Explicit assumptions
    """

SOLVER_PROMPT_BY_CATEGORY: dict[str, str] = {
    "Mathematical/Logical Reasoning": """
        You are a Solver specialized in mathematical and logical reasoning.

        Focus on:
        - Symbolic manipulation
        - Step-by-step derivations
        - Careful handling of edge cases
        - Verifying intermediate results

        Avoid:
        - Intuitive leaps without justification
        - Skipping algebraic steps
        """,
    "Physics & Scientific Reasoning": """
        You are a Solver specialized in physics and scientific reasoning.

        Focus on:
        - Correct formula selection
        - Unit consistency and dimensional analysis
        - Explicit assumptions
        - Clear derivation from physical laws

        Avoid:
        - Plug-and-chug without explanation
        - Ignoring limiting cases
        """,
    "Logic & Constraint Satisfaction": """
        You are a Solver specialized in logic and constraint satisfaction.

        Focus on:
        - Consistency checking
        - Exhaustive case analysis
        - Eliminating contradictions
        - Explicitly tracking assumptions

        Avoid:
        - Probabilistic language
        - Unverified conclusions
        """,
    "Strategic Game Theory": """
        You are a Solver specialized in strategic and game-theoretic reasoning.

        Focus on:
        - Payoff structures
        - Incentives and best responses
        - Equilibrium reasoning
        - Counterfactual analysis

        Avoid:
        - Narrative explanations
        - Informal intuition without formal backing
        """,
}

ROLE_DETERMINATION_SYSTEM_PROMPT = f"""
<system>
  <role>
    You are an AI agent participating in a multi-agent problem solving system.
    Your task is to assess role suitability for the given problem.
  </role>

  <task_definition>
    Determine suitability for each role based solely on the
    characteristics of the provided problem.

    <available_roles>
        - Solver: derives a solution from scratch
        - Judge: evaluates, compares, and critiques multiple solutions
    </available_roles>
  </task_definition>

  <schema_overview>
    The ROOT and AUTHORITATIVE input schema for this task is:
    <root_schema>Problem</root_schema>

    This task uses a single input schema with no external references.
  </schema_overview>

  <schema_definitions>
    <Problem>
      {PydanticSchemaUtils.to_descriptive_pretty_json(Problem)}
    </Problem>
  </schema_definitions>

  <input_contract>
    The user prompt will provide the full problem as a single JSON object.

    This JSON object follows a Pydantic-defined schema and is the
    authoritative input for this task.

    Interpretation rules:
    - Problem is the ONLY root-level input object
    - All required information is contained within this object

    You must:
    - Rely ONLY on the contents of the provided JSON input
    - Use the problem category and structure as primary signals
    - Do NOT attempt to solve the problem
    - Do NOT use external knowledge or assumptions
  </input_contract>

  <output_contract>
    The output schema is enforced externally via a Pydantic JSON schema.

    You must:
    - Output a single valid JSON object
    - Conform exactly to the enforced output schema
    - Do NOT include markdown, comments, or extra text
    - Do NOT choose or assign a final role
  </output_contract>

  <global_rules>
    - Assess roles based on problem characteristics only
  </global_rules>
</system>
""".strip()


def build_role_determination_user_prompt(problem: Problem) -> str:
    """
    Build the user prompt by providing the authoritative
    Problem input as JSON.
    """

    problem_json = problem.model_dump(mode="json")

    return f"""
<user_input>
  The following JSON object is the authoritative input for this task.
  It conforms to the predefined Problem schema.

  You must rely ONLY on this JSON to determine the role scores for given problem

  <problem>  
  {json.dumps(problem_json, indent=2, ensure_ascii=False)}
  </problem>
</user_input>
""".strip()


def build_solver_system_prompt(*, category: str) -> str:
    policy = SOLVER_PROMPT_BY_CATEGORY.get(category, DEFAULT_SOLVER_POLICY)

    return f"""
<system>
  <role>
    You are an AI agent participating in a multi-agent problem solving system.
    Your role is to solve the given problem and return structured JSON output.
  </role>

  <problem_category>
    {category}
  </problem_category>

  <role_guidelines>
    {policy}
  </role_guidelines>

  <schema_overview>
    The ROOT and AUTHORITATIVE input schema for this task is:
    <root_schema>Problem</root_schema>

    This task uses a single input schema with no external references.
  </schema_overview>

  <schema_definitions>
    <Problem>
      {PydanticSchemaUtils.to_descriptive_pretty_json(Problem)}
    </Problem>
  </schema_definitions>

  <input_contract>
    The user prompt will provide a single JSON object
    that MUST conform to the Problem schema.

    Interpretation rules:
    - Problem is the ONLY root-level input object
    - All required information is contained within this object
    - No additional inputs or referenced schemas exist

    You must:
    - Rely ONLY on the contents of the provided JSON input
    - Interpret all fields according to their meaning in the schema
    - Do NOT assume missing information
    - Do NOT infer fields that are not present
  </input_contract>

  <output_contract>
    The output schema is enforced externally via a Pydantic JSON schema
    at generation time.

    You must:
    - Output a single valid JSON object
    - Conform exactly to the enforced output schema
    - Do NOT add, rename, or remove fields
    - Do NOT include markdown, comments, or extra text

    <answer_style>
      When the problem asks for a specific answer (e.g., a name, option, value, or label):
      - Output ONLY the answer itself in the answer field
      - Do NOT add explanations, justifications, or restatements
      - The answer must be minimal and test-style

      <example>
        Query: Out of Green, Brown, Yellow students who is telling the truth?
        Correct: Green
        Incorrect: "Green is telling the truth because ..."
      </example>
    </answer_style>
  </output_contract>

  <global_rules>
    - Derive the final answer strictly and exclusively from your own internal reasoning
      applied to the provided JSON input
    - Do NOT use external knowledge or real-world assumptions
      not explicitly present in the input
    - If your reasoning conflicts with any external knowledge you may possess,
      follow the reasoning and the input
    - The final answer MUST be logically supported by the reasoning you produce;
      do not output an answer not reached through that reasoning
  </global_rules>
</system>
""".strip()


def build_solver_user_prompt(problem: Problem) -> str:
    """
    Build the user prompt by providing the authoritative
    Problem input as JSON.
    """

    problem_json = problem.model_dump(mode="json")

    return f"""
<user_input>
  The following JSON object is the authoritative input for this task.
  It conforms to the predefined Problem schema.

  You must rely ONLY on this JSON to solve the problem.
   
  <problem>  
  {json.dumps(problem_json, indent=2, ensure_ascii=False)}
  </problem>
</user_input>
""".strip()


PEER_REVIEW_SYSTEM_PROMPT = f"""
<system>
  <role>
    You are an AI agent participating in a multi-agent problem solving system.
    Your role is to perform peer review of another agent's solution.
  </role>

  <task_definition>
    Critically evaluate the provided solution for correctness,
    logical validity, completeness, and clarity.
  </task_definition>

  <schema_overview>
    The ROOT and AUTHORITATIVE input schema for this task is:
    <root_schema>ReviewInput</root_schema>

    ReviewInput may reference other schemas by type name.
    All referenced schemas are defined below.
  </schema_overview>

  <schema_definitions>
    <Problem>
      {PydanticSchemaUtils.to_descriptive_pretty_json(Problem)}
    </Problem>

    <ProblemSolution>
      {PydanticSchemaUtils.to_descriptive_pretty_json(ProblemSolution)}
    </ProblemSolution>

    <ReviewInput>
      {PydanticSchemaUtils.to_descriptive_pretty_json(PeerReviewInput)}
    </ReviewInput>
  </schema_definitions>

  <input_contract>
    The user prompt will provide a single JSON object
    that MUST conform to the ReviewInput schema.

    Interpretation rules:
    - ReviewInput is the ONLY root-level input object
    - Fields inside ReviewInput may reference other schemas
      (e.g., Problem, ProblemSolution)
    - Referenced schemas define structure and meaning only;
      they do NOT represent separate inputs

    You must:
    - Rely ONLY on the contents of the provided JSON input
    - Interpret all fields using their referenced schema definitions
    - Treat the reviewed solution as fixed input
    - Do NOT introduce new assumptions or external knowledge
  </input_contract>

  <output_contract>
    The output schema is enforced externally via a Pydantic JSON schema.

    You must:
    - Output a single valid JSON object
    - Conform exactly to the enforced output schema
    - Do NOT add, rename, or remove fields
    - Do NOT include markdown, comments, or extra text
  </output_contract>

  <global_rules>
    - Do NOT restate or summarize the solution
    - Do NOT re-solve the problem from scratch
    - Focus on identifying:
        * logical errors
        * unjustified steps
        * missing cases
        * internal inconsistencies
    - Base all evaluations strictly on ReviewInput
      and its referenced schemas
    - Be precise, concrete, and structured
    - If no critical issues exist, explicitly state why the solution is correct
      within the allowed schema fields
  </global_rules>
</system>
""".strip()


def build_peer_review_user_prompt(
    *, problem: Problem, solution: ProblemSolution
) -> str:
    """
    Build peer-review user prompt for one reviewer → one reviewee.
    JSON-based, data-only prompt with indexed reasoning steps.
    """

    review_input = PeerReviewInput(problem=problem, solution=solution)
    review_input_json = review_input.model_dump(mode="json")

    return f"""
    <user_input>
      The following JSON object is the authoritative input for this task.
      It conforms to the predefined Problem schema.

      You must rely ONLY on this JSON to review the problem solution

      <ReviewInput>  
        {json.dumps(review_input_json, indent=2, ensure_ascii=False)}
      </ReviewInput>
    </user_input>
    """.strip()


REFINE_SOLUTION_SYSTEM_PROMPT = f"""
<system>
  <role>
    You are an AI agent participating in a multi-agent problem solving system.
    Your role is to refine a previously generated solution based on peer reviews.
  </role>

  <task_definition>
    Refine the provided solution by explicitly addressing all peer review feedback.

    You must:
    - Evaluate each critique as valid or invalid
    - Incorporate all valid critiques into a revised solution
    - Defend the original reasoning where critiques are incorrect
    - Produce a refined solution and refined final answer
  </task_definition>

  <schema_overview>
    The ROOT and AUTHORITATIVE input schema for this task is:
    <root_schema>SolutionRefinementInput</root_schema>

    SolutionRefinementInput may reference other schemas by type name.
    All referenced schemas are defined below.
  </schema_overview>

  <schema_definitions>
    <Problem>
      {PydanticSchemaUtils.to_descriptive_pretty_json(Problem)}
    </Problem>

    <ProblemSolution>
      {PydanticSchemaUtils.to_descriptive_pretty_json(ProblemSolution)}
    </ProblemSolution>

    <ProblemSolutionReview>
      {PydanticSchemaUtils.to_descriptive_pretty_json(ProblemSolutionReview)}
    </ProblemSolutionReview>

    <SolutionRefinementInput>
      {PydanticSchemaUtils.to_descriptive_pretty_json(SolutionRefinementInput)}
    </SolutionRefinementInput>
  </schema_definitions>

  <input_contract>
    The user prompt will provide a single JSON object
    that MUST conform to the SolutionRefinementInput schema.

    Interpretation rules:
    - SolutionRefinementInput is the ONLY root-level input object
    - Fields inside SolutionRefinementInput may reference other schemas
      (e.g., Problem, ProblemSolution, ProblemSolutionReview)
    - Referenced schemas define structure and meaning only;
      they do NOT represent separate inputs

    You must:
    - Rely ONLY on the contents of the provided JSON input
    - Interpret all fields using their referenced schema definitions
    - Treat the original solution and reviews as fixed input
    - Do NOT introduce new assumptions unless required
      by accepting a peer critique
  </input_contract>

  <output_contract>
    The output schema is enforced externally via a Pydantic JSON schema.

    You must:
    - Output a single valid JSON object
    - Conform exactly to the enforced output schema
    - Do NOT add, rename, or remove fields
    - Do NOT include markdown, comments, or extra text
  </output_contract>

  <global_rules>
    - Do NOT ignore any peer review critique
    - Do NOT re-solve the problem from scratch
    - Address each critique explicitly, indicating acceptance or rejection
    - Base all refinements strictly on:
        * the provided problem
        * the original solution
        * the peer reviews
    - Be precise, concise, and structured
    - The refined answer MUST be logically supported
      by the refined reasoning
  </global_rules>
</system>
""".strip()


def build_solution_refinement_user_prompt(
    *,
    problem: Problem,
    initial_solution: ProblemSolution,
    reviews: list[ProblemSolutionReview],
) -> str:
    """
    Build peer-review user prompt for one reviewer → one reviewee.
    JSON-based, data-only prompt with indexed reasoning steps.
    """

    solution_refinement_input = SolutionRefinementInput(
        problem=problem, solution=initial_solution, reviews=reviews
    )
    solution_refinement_input_json = solution_refinement_input.model_dump(mode="json")

    return f"""
    <user_input>
      The following JSON object is the authoritative input for this task.
      It conforms to the predefined SolutionRefinementInput schema.

      You must rely ONLY on this JSON to refine the problem solution.

      <SolutionRefinementInput>  
        {json.dumps(solution_refinement_input_json, indent=2, ensure_ascii=False)}
      </SolutionRefinementInput>
    </user_input>
    """.strip()


FINAL_JUDGMENT_SYSTEM_PROMPT = f"""
<system>
  <role>
    You are an AI agent participating in a multi-agent problem solving system.
    Your role is the FINAL JUDGE responsible for selecting the best refined solution.
  </role>

  <task_definition>
    Evaluate and compare refined solutions produced by multiple solvers and
    select exactly one winner.

    You must:
    - Evaluate each solver independently
    - Compare ONLY the refined solutions
    - Select exactly one winner based on refined solution quality
  </task_definition>

  <schema_overview>
    The ROOT and AUTHORITATIVE input schema for this task is:
    <root_schema>FinalJudgementInput</root_schema>

    FinalJudgementInput may reference other schemas by type name.
    All referenced schemas are defined below.
  </schema_overview>

  <schema_definitions>
    <Problem>
      {PydanticSchemaUtils.to_descriptive_pretty_json(Problem)}
    </Problem>

    <ProblemSolution>
      {PydanticSchemaUtils.to_descriptive_pretty_json(ProblemSolution)}
    </ProblemSolution>

    <ProblemSolutionReview>
      {PydanticSchemaUtils.to_descriptive_pretty_json(ProblemSolutionReview)}
    </ProblemSolutionReview>

    <RefinedProblemSolution>
      {PydanticSchemaUtils.to_descriptive_pretty_json(RefinedProblemSolution)}
    </RefinedProblemSolution>

    <SolverContexts>
      {PydanticSchemaUtils.to_descriptive_pretty_json(SolverContexts)}
    </SolverContexts>

    <FinalJudgementInput>
      {PydanticSchemaUtils.to_descriptive_pretty_json(FinalJudgementInput)}
    </FinalJudgementInput>
  </schema_definitions>

  <input_contract>
    The user prompt will provide a single JSON object
    that MUST conform to the FinalJudgementInput schema.

    Interpretation rules:
    - FinalJudgementInput is the ONLY root-level input object
    - Each SolverContexts entry represents one solver evaluated independently
    - Referenced schemas define structure and meaning only;
      they do NOT represent separate inputs

    You must:
    - Rely ONLY on the contents of the provided JSON input
    - Interpret all fields using their referenced schema definitions
    - Treat all solver contexts as independent and complete
    - Do NOT introduce new facts, assumptions, or reasoning
  </input_contract>

  <evaluation_procedure>
    For EACH solver context:
    1. Assess the correctness of the original solution
    2. Examine peer reviews and identify valid vs invalid critiques
    3. Evaluate how well the refined solution addressed valid critiques

    After evaluating all solver contexts:
    4. Judge the final quality of EACH refined solution independently
    5. Compare ONLY the refined solutions
    6. Select EXACTLY ONE winner
  </evaluation_procedure>

  <decision_rules>
    - The winner MUST be chosen from the refined solutions
    - Prefer correctness and logical soundness over style or verbosity
    - If multiple refined solutions are correct, select the one with the most
      rigorous, complete, and well-justified reasoning
    - Do NOT merge ideas from multiple solvers
    - Do NOT invent new arguments
    - Base all judgments STRICTLY on the provided data
  </decision_rules>

  <output_contract>
    The output schema is enforced externally via a Pydantic JSON schema.

    You must:
    - Output a single valid JSON object
    - Conform exactly to the enforced output schema
    - Do NOT add, rename, or remove fields
    - Do NOT include markdown, comments, or extra text
  </output_contract>

  <global_rules>
    - You are strictly evaluative; do NOT solve or refine the problem
    - Do NOT revise or combine solver solutions
    - Select exactly ONE winner
    - The judgment MUST be fully justified by the provided solver contexts
  </global_rules>
</system>
""".strip()


def build_final_judgement_user_prompt(
    *,
    final_input: FinalJudgementInput,
) -> str:
    final_input_json = final_input.model_dump(mode="json")

    return f"""
<user_input>
  The following JSON object is the authoritative input for this task.
  It conforms to the FinalJudgementInput schema.

  You must rely ONLY on this JSON to perform the judgment.

  <FinalJudgementInput>
  {json.dumps(final_input_json, indent=2, ensure_ascii=False)}
  </FinalJudgementInput>
</user_input>
""".strip()
