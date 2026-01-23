import asyncio
from typing import Dict, List, Any

from dotenv import load_dotenv

from data.persistence.firestore_client import get_firestore_client
from data.persistence.firestore_manager import (
    FirestoreManager,
    RUNS,
    ROLE_ASSESSMENTS,
    SOLUTIONS,
    SOLUTION_REVIEWS,
    REFINED_SOLUTIONS,
    FINAL_JUDGEMENTS,
    METRICS,
)


async def main() -> None:
    load_dotenv()

    # ---------------------------------
    # Firestore setup
    # ---------------------------------
    db = get_firestore_client()
    firestore = FirestoreManager(db)

    # ---------------------------------
    # Dump all collections
    # ---------------------------------
    runs: List[Dict[str, Any]] = await firestore.dump_collection(collection=RUNS)
    role_assessments: List[Dict[str, Any]] = await firestore.dump_collection(
        collection=ROLE_ASSESSMENTS
    )
    solutions: List[Dict[str, Any]] = await firestore.dump_collection(
        collection=SOLUTIONS
    )
    solution_reviews: List[Dict[str, Any]] = await firestore.dump_collection(
        collection=SOLUTION_REVIEWS
    )
    refined_solutions: List[Dict[str, Any]] = await firestore.dump_collection(
        collection=REFINED_SOLUTIONS
    )
    final_judgements: List[Dict[str, Any]] = await firestore.dump_collection(
        collection=FINAL_JUDGEMENTS
    )
    metrics: List[Dict[str, Any]] = await firestore.dump_collection(
        collection=METRICS
    )

    # ---------------------------------
    # Debug / sanity prints
    # ---------------------------------
    print(f"Runs: {len(runs)}")
    print(f"RoleAssessments: {len(role_assessments)}")
    print(f"Solutions: {len(solutions)}")
    print(f"SolutionReviews: {len(solution_reviews)}")
    print(f"RefinedSolutions: {len(refined_solutions)}")
    changed_refinements = list(
        filter(lambda r: r.get("answer_changed") is True, refined_solutions)
    )

    print("Refinements with changed answer:", len(changed_refinements))
    print(f"FinalJudgements: {len(final_judgements)}")
    print(f"Metrics: {len(metrics)}")


if __name__ == "__main__":
    asyncio.run(main())
