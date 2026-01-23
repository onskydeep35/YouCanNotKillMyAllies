import asyncio
from typing import Dict, List, Any

import pandas as pd
from dotenv import load_dotenv

from data.persistence.firestore_client import get_firestore_client
from data.persistence.firestore_manager import (
    FirestoreManager,
    RUNS,
    PROBLEMS,
    SOLUTIONS,
    REFINED_SOLUTIONS,
    FINAL_JUDGEMENTS,
    METRICS,
)


def project(records: List[Any], columns: List[str]) -> pd.DataFrame:
    rows = []
    for r in records:
        if not isinstance(r, dict):
            continue
        row = {}
        for c in columns:
            row[c] = r.get(c)
        rows.append(row)
    return pd.DataFrame(rows)


async def main() -> None:
    load_dotenv()

    db = get_firestore_client()
    firestore = FirestoreManager(db)

    # -------------------------------
    # Load collections (single read)
    # -------------------------------
    problems = await firestore.dump_collection(collection=PROBLEMS)
    solutions = await firestore.dump_collection(collection=SOLUTIONS)
    refined_solutions = await firestore.dump_collection(collection=REFINED_SOLUTIONS)
    final_judgements = await firestore.dump_collection(collection=FINAL_JUDGEMENTS)

    # -------------------------------
    # SAFE normalization (NO crashes)
    # -------------------------------
    df_problems = project(
        problems,
        ["problem_id", "ground_answer"]
    )

    df_solutions = project(
        solutions,
        ["problem_id", "answer"]
    )

    df_refined = project(
        refined_solutions,
        ["problem_id", "answer_changed"]
    )

    df_final = project(
        final_judgements,
        ["problem_id", "answer", "confidence", "time_elapsed_sec"]
    )

    # -------------------------------
    # Join ground truth
    # -------------------------------
    df_final = df_final.merge(
        df_problems,
        on="problem_id",
        how="left"
    )

    df_final["is_correct"] = df_final["answer"] == df_final["ground_answer"]

    # -------------------------------
    # SYSTEM-LEVEL METRICS
    # -------------------------------
    metrics: Dict[str, Any] = {}

    metrics["overall_accuracy"] = float(df_final["is_correct"].mean())

    solver_answers = (
        df_solutions
        .groupby("problem_id")["answer"]
        .apply(list)
    )

    consensus = solver_answers.apply(lambda x: len(set(x)) == 1)
    disagreement = solver_answers.apply(lambda x: len(set(x)) > 1)

    metrics["consensus_rate"] = float(consensus.mean())
    metrics["disagreement_rate"] = float(disagreement.mean())

    if not df_refined.empty:
        refinement_changed = (
            df_refined
            .groupby("problem_id")["answer_changed"]
            .any()
        )
        final_correct = df_final.set_index("problem_id")["is_correct"]
        improvement_mask = refinement_changed & final_correct
        metrics["improvement_rate"] = float(improvement_mask.mean())
    else:
        metrics["improvement_rate"] = 0.0

    disagreement_ids = set(disagreement[disagreement].index)

    if disagreement_ids:
        judge_subset = df_final[df_final["problem_id"].isin(disagreement_ids)]
        metrics["judge_accuracy_on_disagreement"] = float(
            judge_subset["is_correct"].mean()
        )
    else:
        metrics["judge_accuracy_on_disagreement"] = 0.0

    metrics["avg_confidence_final"] = float(
        df_final["confidence"].mean()
    ) if "confidence" in df_final else None

    metrics["avg_time_to_final"] = float(
        df_final["time_elapsed_sec"].mean()
    ) if "time_elapsed_sec" in df_final else None

    # -------------------------------
    # Persist metrics
    # -------------------------------
    await firestore.write(
        collection=METRICS,
        document={
            "metric_type": "system_level",
            "metrics": metrics,
        },
    )

    print("SYSTEM METRICS")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
