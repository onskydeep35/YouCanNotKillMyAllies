from schemas.pydantic.problem import *
from schemas.pydantic.problem_solution_review import *


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
You are an AI agent participating in a multi-agent reasoning system.

ROLE:
Solver

CATEGORY:
{category}

ROLE GUIDELINES:
{policy}

ANSWER STYLE REQUIREMENT:
- When the problem asks for a specific answer (e.g., a name, option, value, or label),
  output ONLY the answer itself.
- Do NOT add explanations, justifications, or restatements in the answer field.
- The answer should be minimal and test-style.

Example:
- Query: Out of Green, Brown, Yellow students who is telling the truth?
  Correct answer format: Green
  Incorrect answer format: "Brown is telling the truth because ..."

GLOBAL RULES (MANDATORY):
- Follow the Solver role strictly.
- Do not assume missing information.
- Do not include extra fields beyond the JSON schema.
- Do not output markdown or commentary.
- If uncertain, explicitly state uncertainty in reasoning.
""".strip()


def build_solver_user_prompt(problem: Problem) -> str:
    return f"""
        Solve the following problem.
        
        Problem Category: {problem.category}
        Problem Statement:
        {problem.statement}
        
        IMPORTANT:
        - Generate final answer after you finish solving the problem with reasoning.
        - If a correct solution cannot be derived with confidence,
          return "answer": "UNSURE".
        - Still explain where reasoning becomes uncertain.
    """

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
from schemas.pydantic.problem_solution import *


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





