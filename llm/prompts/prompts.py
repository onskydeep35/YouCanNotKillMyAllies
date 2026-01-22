from schemas.pydantic.input.problem import *
from schemas.pydantic.output.problem_solution_review import *
from schemas.utilities.pydantic_schema_utils import PydanticSchemaUtils

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


ROLE_DETERMINATION_SYSTEM_PROMPT = """
You are an AI agent participating in a multi-agent collaborative reasoning system.

TASK:
Assess suitability for different roles based on the PROBLEM CHARACTERISTICS,
not personal preference.

ROLES:
- Solver: derives a solution from scratch.
- Judge: evaluates, critiques, and compares multiple solutions.

GUIDELINES:
- Use the problem category as a strong prior.
- Think in terms of task difficulty, verification needs, and reasoning style.
- Do NOT choose a final role.
- Do NOT include explanations beyond what is required by the schema.
- Output must strictly follow the provided JSON schema.
"""


def build_role_determination_user_prompt(problem: Problem) -> str:
    return f"""
Analyze the following problem and estimate how suitable each role is.

Problem Category: {problem.category}

Problem Statement:
{problem.statement}

Return normalized confidence scores between 0.0 and 1.0 for each role.
"""


def build_solver_system_prompt(*, category: str) -> str:
    policy = SOLVER_PROMPT_BY_CATEGORY.get(category, DEFAULT_SOLVER_POLICY)

    return f"""
<system>
  <role>
    You are an AI agent participating in a multi-agent reasoning system.
    Your role is to solve the given problem and return structured JSON output.
  </role>

  <problem_category>
    {category}
  </problem_category>

  <role_guidelines>
    {policy}
  </role_guidelines>

  <input_contract>
    The user prompt will provide the full problem and all required inputs
    as a single JSON object.
    
    This JSON object follows a Pydantic-defined schema and represents
    the authoritative input for this task.
    
    <schema_description>
      {PydanticSchemaUtils.to_descriptive_pretty_json(Problem)}
    </schema_description>
    
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
  </output_contract>

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

  <global_rules>
    - Follow the Solver role strictly
    - Do not assume missing information
    - Do not output markdown or commentary
    - If uncertain, explicitly state uncertainty ONLY in the designated schema fields
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
   
  <problem_json>  
  {json.dumps(problem_json, indent=2, ensure_ascii=False)}
  </problem_json>
</user_input>
""".strip()


PEER_REVIEW_SYSTEM_PROMPT = """
You are an expert peer reviewer evaluating another AI's solution.

Your task is to critically evaluate the solution for correctness, logical validity,
completeness, and clarity.

Rules:
- Do NOT restate the solution.
- Do NOT re-solve the problem from scratch.
- Focus on identifying flaws, missing cases, unjustified steps, or inconsistencies.
- Be precise and structured.
- If the solution is correct, explicitly state why no critical errors exist.
- Output ONLY valid JSON matching the given schema.
""".strip()

import json
from schemas.pydantic.output.problem_solution import *


def build_peer_review_user_prompt(
    *,
    problem: Problem,
    solution: ProblemSolution
) -> str:
    """
    Build peer-review user prompt for one reviewer → one reviewee.
    JSON-based, data-only prompt with indexed reasoning steps.
    """

    payload = {
        "problem": {
            "statement": problem.statement
        },
        "solution": {
            "answer": solution.answer,
            "reasoning_steps": [
                {
                    "step": idx + 1,
                    "text": step
                }
                for idx, step in enumerate(solution.reasoning)
            ]
        }
    }

    user_prompt = (
        "Below is the data required to review the solution.\n"
        "Use ALL information provided.\n\n"
        "INPUT_DATA:\n"
        f"{json.dumps(payload, indent=2)}"
    )

    print(f"[REVIEW USER PROMPT]: {user_prompt}")
    return user_prompt


REFINE_SOLUTION_SYSTEM_PROMPT = """
You are a problem solution refiner.

You previously produced a solution to a problem.
You are now given peer reviews written by other agents.

Your task:
- Explicitly address every critique found in the peer reviews
- Decide whether each critique is valid or invalid
- Incorporate all valid critiques into a revised solution
- Defend your original reasoning when critiques are incorrect
- Produce a refined final solution and final answer

Rules:
- Do NOT ignore any critique
- Do NOT introduce new assumptions unless required by accepted critiques
- Be precise and concise
- Base all revisions strictly on the provided problem, solution, and reviews
"""


def build_solution_refinement_user_prompt(
    *,
    problem: Problem,
    initial_solution: ProblemSolution,
    reviews: list[ProblemSolutionReview],
) -> str:
    """
    Builds the user prompt for Stage 3 solution refinement.
    Input is intentionally JSON to preserve full structure.
    """

    problem_payload = problem.model_dump()

    initial_solution_payload = initial_solution.model_dump()

    reviews_payload = [
        review.model_dump()
        for review in reviews
    ]

    payload = {
        "problem": problem_payload,
        "previous_solution": initial_solution_payload,
        "peer_reviews": reviews_payload,
    }

    user_prompt = (
        "Below is the data required to refine the solution.\n"
        "Use ALL information provided.\n\n"
        "INPUT_DATA:\n"
        f"{json.dumps(payload, indent=2)}"
    )

    print(f"[REFINEMENT USER PROMPT]: {user_prompt}")

    return user_prompt


FINAL_JUDGMENT_SYSTEM_PROMPT = """
You are the FINAL JUDGE in a structured multi-agent problem-solving debate.

Your role is strictly evaluative and comparative.
You do NOT solve the problem yourself.
You do NOT revise solutions.
You do NOT introduce new facts, assumptions, or reasoning.

You will receive ALL input as structured JSON payloads, grouped by entity.
Each solver is presented independently and must be evaluated independently.

You are given:
- The original problem
- Three solver sections (Solver 1, Solver 2, Solver 3)
For EACH solver, you will receive:
- The original solution
- All peer reviews written about that solution
- The refined solution produced after incorporating peer feedback

Your evaluation procedure MUST be:
1. For each solver:
   - Assess the correctness of the original solution
   - Examine peer reviews and identify valid vs invalid critiques
   - Evaluate how well the refined solution addressed valid critiques
2. Judge the final quality of EACH refined solution independently
3. Compare ONLY the refined solutions to each other
4. Select EXACTLY ONE winner based on the refined solutions

Decision rules:
- The winner MUST be chosen from the refined solutions
- Prefer correctness and logical soundness over style or verbosity
- If multiple refined solutions are correct, select the one with most rigorous, complete, and well-justified reasoning
- Do NOT merge ideas from multiple solvers
- Do NOT invent new arguments
- Base your judgment STRICTLY on the provided data

Output requirements:
- Choose exactly ONE winner: "Solver 1", "Solver 2", or "Solver 3"
- Provide step-by-step reasoning justifying the decision
- Assign a confidence score between 0.0 and 1.0
"""



