import asyncio
from typing import Dict, Any, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from data.persistence.firestore_client import get_firestore_client
from data.persistence.firestore_manager import (
    FirestoreManager,
    PROBLEMS,
    FINAL_JUDGEMENTS, SOLUTIONS, REFINED_SOLUTIONS,
)

from llm.agents.agent_factory import AgentFactory
from schemas.dataclass.agent_config import LLMAgentConfig


# =================================================
# SCHEMAS
# =================================================
class AnswerComparisonInput(BaseModel):
    problem: str
    ground_answer: str
    solution_answer: str


class AnswerCorrectnessJudgement(BaseModel):
    is_correct: bool = Field(...)


# =================================================
# PROMPTS
# =================================================
def build_correctness_system_prompt() -> str:
    return """
<system_contract>
  <role>
    You are a strict answer-equivalence verification engine.
    Your task is NOT to solve the problem.
  </role>

  <scope>
    Compare a ground-truth answer with a proposed solution answer
    and determine whether they represent the same final result.
  </scope>

  <rules>
    <rule>Do NOT attempt to solve or reason about the problem.</rule>
    <rule>Compare ONLY the ground_answer and solution_answer.</rule>
    <rule>Ignore differences in formatting, symbols, or notation.</rule>
    <rule>Mathematically equivalent expressions are considered equal.</rule>
    <rule>Examples of equivalent answers include:
      <example>pi/6 == π/6</example>
      <example>499500 == 499,500</example>
      <example>0.5 == 1/2</example>
    </rule>
    <rule>If the answers cannot be confidently determined to be equivalent, return false.</rule>
    <rule>Return ONLY a single valid JSON object.</rule>
    <rule>No explanations, reasoning, or extra fields.</rule>
  </rules>

  <output_schema>
    { "is_correct": boolean }
  </output_schema>
</system_contract>
""".strip()



def build_correctness_user_prompt(inp: AnswerComparisonInput) -> str:
    return f"""
<problem>
{inp.problem}
</problem>

<ground_answer>
{inp.ground_answer}
</ground_answer>

<solution_answer>
{inp.solution_answer}
</solution_answer>
""".strip()


# =================================================
# PER-JUDGEMENT WORKER
# =================================================
async def process_judgement(
    *,
    judgement: Dict[str, Any],
    problem_map: Dict[str, Dict[str, str]],
    agent,
    firestore: FirestoreManager,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    async with semaphore:
        try:
            problem_id = judgement.get("problem_id")
            judgement_id = judgement.get("judgement_id")  # NOW GUARANTEED
            answer = judgement.get("refined_answer")

            if not problem_id or not judgement_id or not answer:
                return False

            problem = problem_map.get(problem_id)
            if not problem:
                return False

            inp = AnswerComparisonInput(
                problem=problem["statement"],
                ground_answer=problem["ground_answer"],
                solution_answer=answer,
            )

            result = await agent.run_structured_call(
                problem=None,
                system_prompt=system_prompt,
                user_prompt=build_correctness_user_prompt(inp),
                output_model=AnswerCorrectnessJudgement,
                method_type="none",
            )

            await firestore.update_document(
                collection=FINAL_JUDGEMENTS,
                document_id=judgement_id,
                updates={
                    "is_correct_answer": result.is_correct,
                },
            )

            return True

        except Exception as e:
            print(f"[SKIP] judgement={judgement.get('judgement_id')} err={e}")
            return False


# =================================================
# MAIN
# =================================================
async def main() -> None:
    load_dotenv()

    db = get_firestore_client()
    firestore = FirestoreManager(db)

    agent = AgentFactory.create_agent(
        config=LLMAgentConfig(
            provider="gemini",
            llm_id="gemini-3-pro-1",
            model="gemini-3-flash-preview",
            temperature=0.6,
            top_p=0.9,
        )
    )

    problems: List[Dict[str, Any]] = await firestore.dump_collection(
        collection=PROBLEMS
    )

    solutions: List[Dict[str, Any]] = await firestore.dump_collection(
        collection=FINAL_JUDGEMENTS
    )

    # ---------------------------------------------
    # Build problem lookup
    # ---------------------------------------------
    problem_map: Dict[str, Dict[str, str]] = {}
    for p in problems:
        pid = p.get("problem_id")
        if pid and "statement" in p and "ground_answer" in p:
            problem_map[pid] = {
                "statement": p["statement"],
                "ground_answer": p["ground_answer"],
            }

    # ---------------------------------------------
    # 🔴 FIX BROKEN JUDGEMENT IDS
    # ---------------------------------------------
    # EXPECTATION:
    # dump_collection() MUST include Firestore doc id as "__id__"
    # If your key is different, change HERE only.
    FIXED = 0

    for j in solutions:
        doc_id = j.get("__id__")  # 🔴 CHANGE IF NEEDED

        if not doc_id:
            continue

        # overwrite broken field in memory
        j["refined_solution_id"] = doc_id

        # persist fix for future sanity
        await firestore.update_document(
            collection=SOLUTIONS,
            document_id=doc_id,
            updates={
                "refined_solution_id": doc_id,
            },
        )

        FIXED += 1

    print(f"Fixed refined_solution_id for {FIXED} documents")

    # ---------------------------------------------
    # Run async correctness judging
    # ---------------------------------------------
    system_prompt = build_correctness_system_prompt()
    semaphore = asyncio.Semaphore(50)

    tasks = [
        process_judgement(
            judgement=j,
            problem_map=problem_map,
            agent=agent,
            firestore=firestore,
            system_prompt=system_prompt,
            semaphore=semaphore,
        )
        for j in solutions
    ]

    results = await asyncio.gather(*tasks)

    updated = sum(1 for r in results if r)
    skipped = len(results) - updated

    print(f"Done. Updated={updated}, Skipped={skipped}")


if __name__ == "__main__":
    asyncio.run(main())
