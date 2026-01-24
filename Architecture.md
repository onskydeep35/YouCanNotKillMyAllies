# Architecture Notes

This document complements the README by summarizing the concrete components and data artifacts in the multi‑LLM debate system.

## Core Components

- **Orchestrator**: `ProblemSolvingApp` loads the dataset, initializes agents once, and runs a `ProblemSolvingSession` per problem with concurrency controls. 【F:runtime/app.py†L1-L113】
- **Session pipeline**: `ProblemSolvingSession` drives the staged workflow and persists every stage to Firestore and local JSON files. 【F:runtime/problem_solving_session.py†L1-L401】
- **Agent contexts**:
  - `SolverAgentContext` encapsulates role assessment, solution generation, peer review, and refinement. 【F:runtime/contexts/solver_agent_context.py†L1-L204】
  - `JudgeAgentContext` aggregates solver artifacts and produces the final judgement. 【F:runtime/contexts/judge_agent_context.py†L1-L169】
- **Provider abstraction**: `AgentFactory` selects provider‑specific agents based on configuration, enabling multi‑provider experiments without changing orchestration logic. 【F:llm/agents/agent_factory.py†L1-L45】

## Data Artifacts

### Firestore collections

The system persists every stage in Firestore using standard collections: Runs, RoleAssessments, Solutions, SolutionReviews, RefinedSolutions, FinalJudgements, and Metrics. 【F:data/persistence/firestore_manager.py†L1-L46】

### Local JSON outputs

Every stage also writes JSON files to `data/output/` for offline analysis, mirroring Firestore documents for reproducibility. 【F:config.py†L17-L21】【F:runtime/problem_solving_session.py†L96-L401】

## Stage-by-Stage Inputs/Outputs

1. **Role assessment**: inputs `Problem`, outputs `RoleAssessment`. 【F:schemas/pydantic/input/problem.py†L1-L44】【F:schemas/pydantic/output/role_assessment.py†L1-L35】
2. **Solve**: inputs `Problem`, outputs `ProblemSolution`. 【F:schemas/pydantic/input/problem.py†L1-L44】【F:schemas/pydantic/output/problem_solution.py†L1-L44】
3. **Peer review**: inputs `Problem` + `ProblemSolution`, outputs `ProblemSolutionReview`. 【F:schemas/pydantic/input/problem.py†L1-L44】【F:schemas/pydantic/output/problem_solution_review.py†L1-L48】
4. **Refinement**: inputs `Problem` + original solution + reviews, outputs `RefinedProblemSolution`. 【F:schemas/pydantic/output/refined_problem_solution.py†L1-L51】
5. **Final judgement**: inputs problem + all solver contexts, outputs `FinalJudgement` with winner metadata. 【F:schemas/pydantic/output/final_judgement.py†L1-L47】【F:runtime/contexts/judge_agent_context.py†L40-L169】

## Configuration Touchpoints

- Default output paths and problem selection are centralized in `config.py`. 【F:config.py†L1-L21】
- Agent provider keys are pulled from environment variables via provider clients. 【F:llm/clients/provider_registry.py†L1-L41】
- Firestore credentials are provided through `FIREBASE_CREDENTIALS`. 【F:data/persistence/firestore_client.py†L1-L42】
README.md
README.md