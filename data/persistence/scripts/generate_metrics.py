import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

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
    ROLE_ASSESSMENTS
)


# -------------------------------------------------
# Schema-aware projection + filtering
# -------------------------------------------------
def project_and_filter(
    records: List[Any],
    *,
    required: List[str],
    optional: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    optional = optional or []
    rows: List[Dict[str, Any]] = []

    for r in records:
        if not isinstance(r, dict):
            continue

        row: Dict[str, Any] = {}

        # required fields
        valid = True
        for field in required:
            if field not in r or r[field] is None:
                valid = False
                break
            row[field] = r[field]

        if not valid:
            continue

        # optional fields
        for field in optional:
            row[field] = r.get(field)

        rows.append(row)

    return rows


# -------------------------------------------------
# Main
# -------------------------------------------------
async def main() -> None:
    load_dotenv()

    db = get_firestore_client()
    firestore = FirestoreManager(db)

    # ---------------------------------------------
    # Dump collections (single read each)
    # ---------------------------------------------
    runs = await firestore.dump_collection(collection=RUNS)
    problems = await firestore.dump_collection(collection=PROBLEMS)
    solutions = await firestore.dump_collection(collection=SOLUTIONS)
    refined_solutions = await firestore.dump_collection(collection=REFINED_SOLUTIONS)
    final_judgements = await firestore.dump_collection(collection=FINAL_JUDGEMENTS)
    role_assesments = await firestore.dump_collection(collection=ROLE_ASSESSMENTS)


    def write_in_file(df: pd.DataFrame, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            out_path,
            sep="\t",
            index=False,
            encoding="utf-8",
        )

    def reshape(df: pd.DataFrame) -> pd.DataFrame:
        # rename only if present
        if "solver_llm_model_id" in df.columns:
            df = df.rename(columns={"solver_llm_model_id": "llm_id"})

        # normalize llm_id values only if column exists
        if "llm_id" in df.columns:
            df["llm_id"] = df["llm_id"].replace({
                "gemini-3-flash-1": "gpt-5-mini",
                "gemini-3-pro-1": "gpt-4.1",
            })

        # drop columns only if they exist
        cols_to_drop = [c for c in ["prompt_user", "prompt_system", "_document_id", "reasoning"] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df

    df_problems = reshape(pd.DataFrame(problems))[["problem_id", "ground_answer"]]
    df_solutions = reshape(pd.DataFrame(solutions))[["run_id", "problem_id", "solution_id", "is_correct_answer", "llm_id", "answer"]]
    df_refined = reshape(pd.DataFrame(refined_solutions))[["run_id", "problem_id", "parent_solution_id", "is_correct_answer", "llm_id", "refined_answer"]]
    df_final = reshape(pd.DataFrame(final_judgements))[["run_id", "problem_id", "is_correct_answer", "llm_id", "winner_solver_id", "answer"]]
    df_role = reshape(pd.DataFrame(role_assesments))[["run_id", "problem_id", "llm_id", "judge_score", "solver_score"]]

    df_join_1 = df_problems.merge(
        df_role,
        on="problem_id",
        how="left",
        suffixes=("", "_role"),
    )
    write_in_file(df_join_1, "../../metrics/debug/join1.tsv")
    write_in_file(df_final, "../../metrics/debug/final.tsv")

    df_join_2 = df_join_1.merge(
        df_solutions,
        on=["problem_id", "run_id", "llm_id"],
        how="inner",
        suffixes=("", "_solution"),
    )

    df_join_3 = df_join_2.merge(
        df_refined,
        on=["problem_id", "run_id", "llm_id"],
        how="inner",
        suffixes=("", "_refined"),
    )

    TOTAL = 22

    df_join_3 = df_join_3.merge(
        df_final[["run_id", "problem_id", "winner_solver_id"]],
        on=["run_id", "problem_id"],
        how="left",
    )

    df_join_3["is_judge_selected_context"] = (
            df_join_3["llm_id"] == df_join_3["winner_solver_id"]
    )

    # optional: drop helper column if you truly want nothing else
    df_join_4 = df_join_3.drop(columns=["winner_solver_id"])
    write_in_file(df_join_4, "../../metrics/debug/grounded.tsv")

    df_llm_summary = (
        df_join_3
        .groupby("llm_id", as_index=False)
        .agg(
            solver_role=("solution_id", "count"),
            correct_answers_count=("is_correct_answer", "sum"),
            refined_correct_answers_count=("is_correct_answer_refined", "sum"),
            judge_selected_agent_count=("is_judge_selected_context", "sum"),
            judge_score_mean=("judge_score", lambda x: round(x.mean(), 2)),
            solver_score_mean=("solver_score", lambda x: round(x.mean(), 2)),
            judge_selected_agent_correct_answer_count=(
                "is_judge_selected_context",
                lambda x: (
                        (x == True)
                        & (df_join_3.loc[x.index, "is_correct_answer_refined"] == True)
                ).sum()
            ),
            true_to_false_change=(
                "is_correct_answer",
                lambda x: (
                        (x == True)
                        & (df_join_3.loc[x.index, "is_correct_answer_refined"] == False)
                ).sum()
            ),
            false_to_true_change=(
                "is_correct_answer",
                lambda x: (
                        (x == False)
                        & (df_join_3.loc[x.index, "is_correct_answer_refined"] == True)
                ).sum()
            ),
        )
    )

    df_llm_summary["total"] = TOTAL
    df_llm_summary["judge_role"] = TOTAL - df_llm_summary["solver_role"]
    write_in_file(df_llm_summary, "../../metrics/agent_level_metrics.tsv")

    TOTAL_PROBLEMS = df_join_4["problem_id"].nunique()

    original_solution_accuracy = df_join_4["is_correct_answer"].mean()

    refined_solution_accuracy = df_join_4["is_correct_answer_refined"].mean()

    judged_solution_accuracy = (
        df_join_4.loc[
            df_join_4["is_judge_selected_context"],
            "is_correct_answer_refined"
        ]
        .mean()
    )

    df_solution_accuracy_metrics = pd.DataFrame([{
        "original_solution_accuracy": round(original_solution_accuracy, 3),
        "refined_solution_accuracy": round(refined_solution_accuracy, 3),
        "judged_solution_accuracy": round(judged_solution_accuracy, 3),
    }])
    write_in_file(df_solution_accuracy_metrics, "../../metrics/overall_accuracy_metrics.tsv")

    true_to_false_rate = (
        (df_join_4["is_correct_answer"] & ~df_join_4["is_correct_answer_refined"])
        .mean()
    )

    false_to_true_rate = (
        (~df_join_4["is_correct_answer"] & df_join_4["is_correct_answer_refined"])
        .mean()
    )

    df_refinement_transition_metrics = pd.DataFrame([{
        "true_to_false_rate": round(true_to_false_rate, 3),
        "false_to_true_rate": round(false_to_true_rate, 3),
    }])
    write_in_file(df_refinement_transition_metrics, "../../metrics/refinement_transition_metrics.tsv")

    consensus_rate3 = (
        df_join_4
        .groupby(["run_id", "problem_id"])["refined_answer"]
        .agg(lambda x: x.nunique() == 1)
        .mean()
    )
    consensus_rate2 = (
        df_join_4
        .groupby(["run_id", "problem_id"])["refined_answer"]
        .agg(lambda x: x.nunique() == 2)
        .mean()
    )
    consensus_rate1 = (
        df_join_4
        .groupby(["run_id", "problem_id"])["refined_answer"]
        .agg(lambda x: x.nunique() == 3)
        .mean()
    )
    df_refinement_transition_metrics = pd.DataFrame([{
        "consensus_rate3": round(consensus_rate3, 3),
        "consensus_rate2": round(consensus_rate2, 3),
        "consensus_rate1": round(consensus_rate1, 3),
    }])
    write_in_file(df_refinement_transition_metrics, "../../metrics/consensus_metrics.tsv")

    judge_eval = (
        df_join_4
        .groupby(
            ["run_id", "problem_id"]
        )[
            ["is_correct_answer_refined", "is_judge_selected_context"]
        ]
        .apply(
            lambda g: pd.Series({
                "has_disagreement": g["is_correct_answer_refined"].nunique() > 1,
                "judge_correct": (
                        g["is_correct_answer_refined"].nunique() > 1
                        and g.loc[
                            g["is_judge_selected_context"],
                            "is_correct_answer_refined"
                        ].any()
                )
            })
        )
    )

    # -------------------------------------------------
    # Aggregate counts + accuracy
    # -------------------------------------------------
    judge_disagreement_cases = int(judge_eval["has_disagreement"].sum())
    judge_correct_cases = int(judge_eval["judge_correct"].sum())

    judge_accuracy = (
        judge_correct_cases / judge_disagreement_cases
        if judge_disagreement_cases > 0
        else 0.0
    )

    # -------------------------------------------------
    # Final metrics table
    # -------------------------------------------------
    df_refinement_transition_metrics = pd.DataFrame([{
        "judge_accuracy_at_disagreement": round(judge_accuracy, 3),
        "judge_correct_cases": judge_correct_cases,
        "judge_disagreement_cases": judge_disagreement_cases,
    }])

    write_in_file(df_refinement_transition_metrics, "../../metrics/judge_accuracy_dissagreement.tsv")


if __name__ == "__main__":
    asyncio.run(main())
