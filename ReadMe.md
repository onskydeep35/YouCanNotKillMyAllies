# YouCanNotKillMyAllies

## 1. Project Overview

This project implements a multi‑LLM debate pipeline that solves structured problems by coordinating several large language models (LLMs), collecting peer critiques, refining solutions, and selecting a final answer through a judge model. It orchestrates multiple providers (OpenAI, Gemini, optionally DeepSeek) to generate diverse solutions, then converges on a best answer through explicit review and judging stages. The motivation for using multiple LLMs is to reduce single‑model blind spots: independent reasoning paths and structured peer review surface errors that a single model can miss, while a dedicated judge enforces comparative evaluation across answers. The main problem it solves is reliability—producing higher‑quality, more defensible solutions than a single model or naive voting by combining diversity, critique, and structured refinement.

## 2. System Architecture & Workflow

### High‑level architecture

- **Orchestrator**: `ProblemSolvingApp` loads problems, constructs agents, and runs sessions concurrently. Each problem spawns a `ProblemSolvingSession`.【F:runtime/app.py†L1-L113】
- **Session pipeline**: `ProblemSolvingSession` executes the multi‑stage workflow (role assignment → solving → peer review → refinement → final judging) and persists results to Firestore and JSON files. 【F:runtime/problem_solving_session.py†L1-L401】
- **Agent contexts**: `SolverAgentContext` and `JudgeAgentContext` encapsulate stage‑specific logic and prompt handling. 【F:runtime/contexts/solver_agent_context.py†L1-L204】【F:runtime/contexts/judge_agent_context.py†L1-L169】
- **Structured schemas**: Pydantic models define the contract for each input/output object, enforcing predictable data shapes across stages. 【F:schemas/pydantic/input/problem.py†L1-L44】【F:schemas/pydantic/output/problem_solution.py†L1-L44】【F:schemas/pydantic/output/problem_solution_review.py†L1-L48】【F:schemas/pydantic/output/refined_problem_solution.py†L1-L51】【F:schemas/pydantic/output/final_judgement.py†L1-L47】

### Workflow phases

1. **Role assignment**
   - Each agent self‑assesses whether it should be a solver or a judge using a structured prompt. The top four agents are selected, and the most judge‑aligned agent becomes the judge. 【F:runtime/problem_solving_session.py†L72-L205】【F:runtime/contexts/solver_agent_context.py†L35-L96】
2. **Independent solution generation**
   - Three solver agents independently produce structured solutions using category‑specific system prompts. 【F:runtime/problem_solving_session.py†L211-L252】【F:runtime/contexts/solver_agent_context.py†L98-L144】【F:llm/prompts/prompts.py†L1-L146】
3. **Peer review**
   - Solvers critique each other’s solutions, producing structured reviews that are stored and later fed back to the original solver. 【F:runtime/problem_solving_session.py†L254-L339】【F:runtime/contexts/solver_agent_context.py†L146-L188】
4. **Refinement**
   - Each solver refines its solution based on received peer reviews, generating a refined answer. 【F:runtime/problem_solving_session.py†L341-L400】【F:runtime/contexts/solver_agent_context.py†L190-L204】
5. **Final judging**
   - The judge evaluates the original solutions, reviews, and refinements, then selects the winning solver and final answer. 【F:runtime/problem_solving_session.py†L402-L401】【F:runtime/contexts/judge_agent_context.py†L40-L169】

### Data flow

- **Input**: problems are loaded from `data/datasets/problems.json`.【F:config.py†L17-L21】【F:data/datasets/problems.json†L1-L205】
- **Processing**: each stage emits structured JSON objects validated by Pydantic schemas and saved to Firestore collections (Runs, RoleAssessments, Solutions, SolutionReviews, RefinedSolutions, FinalJudgements).【F:data/persistence/firestore_manager.py†L1-L46】【F:runtime/problem_solving_session.py†L78-L401】
- **Outputs**: per‑stage JSON artifacts are written to `data/output/` subfolders for offline analysis. 【F:config.py†L17-L21】【F:runtime/problem_solving_session.py†L96-L401】

## 3. Design Decisions & Patterns

- **Pipeline / staged processing**
  - The debate is broken into explicit stages (role assessment, solve, review, refine, judge) so each step is auditable and can be measured independently. This reduces errors by forcing checkpoints and reduces debugging complexity. 【F:runtime/problem_solving_session.py†L72-L401】
- **Strategy pattern (solver & judge behaviors)**
  - Agent behavior depends on provider/model configuration and role (solver vs. judge). The `AgentFactory` selects provider‑specific classes, while prompts define role‑specific behavior, allowing model strategies to be swapped without changing orchestration logic. 【F:llm/agents/agent_factory.py†L1-L45】【F:llm/prompts/prompts.py†L1-L146】
- **Separation of concerns**
  - Orchestration (`ProblemSolvingApp`), session logic (`ProblemSolvingSession`), agent behaviors (contexts), and persistence (Firestore) are isolated, which improves maintainability and testability. 【F:runtime/app.py†L1-L113】【F:runtime/problem_solving_session.py†L1-L401】【F:runtime/contexts/solver_agent_context.py†L1-L204】【F:data/persistence/firestore_manager.py†L1-L46】
- **Deterministic orchestration vs. model autonomy**
  - The system enforces a deterministic sequence of stages and schema‑based outputs, while models retain autonomy within each stage’s prompt to reason freely. This balances reproducibility with model flexibility. 【F:runtime/problem_solving_session.py†L72-L401】【F:llm/prompts/prompts.py†L1-L146】
- **Structured schemas (typed JSON / Pydantic models)**
  - Every input/output is validated against Pydantic models to ensure consistent data, prevent malformed responses, and allow automated downstream processing (reviews, judging, metrics). 【F:schemas/pydantic/input/problem.py†L1-L44】【F:schemas/pydantic/output/problem_solution.py†L1-L44】【F:schemas/pydantic/output/problem_solution_review.py†L1-L48】【F:schemas/pydantic/output/refined_problem_solution.py†L1-L51】【F:schemas/pydantic/output/final_judgement.py†L1-L47】

## 4. Evaluation & Metrics

Evaluation is performed by analyzing persisted outputs and (optionally) computed metrics stored in Firestore’s `Metrics` collection. The metrics pipeline is intended to compare multi‑agent performance against single‑LLM baselines and simple voting. A helper script can dump existing collections for analysis. 【F:data/persistence/firestore_manager.py†L1-L46】【F:data/persistence/scripts/generate_metrics.py†L1-L67】

Typical metrics tracked from the stored artifacts include:

- **Accuracy**: fraction of problems where the final judged answer matches the dataset’s ground truth.
- **Improvement rate**: fraction of refined solutions that improve over initial solver answers (change detection is already logged in refined solution outputs). 【F:schemas/pydantic/output/refined_problem_solution.py†L1-L51】
- **Consensus rate**: agreement between solver answers or between solvers and the judge.
- **Baseline comparisons**: compare the judged winner against a single model’s original answer and a simple majority vote of solvers.

Generated plots (typically produced from the Metrics collection) provide insights such as: per‑category accuracy, improvement rates across stages, and how often the judge overturns initial solver preference. These plots reveal where the debate pipeline adds value relative to single‑LLM or voting baselines. 【F:data/persistence/firestore_manager.py†L1-L46】

## 5. How Goals Were Achieved

1. **Multiple LLMs are orchestrated** by configuring heterogeneous agent configs in `main.py` and instantiating them through the `AgentFactory`. 【F:main.py†L1-L79】【F:llm/agents/agent_factory.py†L1-L45】
2. **Role specialization** is enforced by the role assessment stage and the solver/judge contexts. 【F:runtime/problem_solving_session.py†L72-L205】【F:runtime/contexts/solver_agent_context.py†L35-L96】【F:runtime/contexts/judge_agent_context.py†L15-L93】
3. **Independent solutions** are generated by three solvers with category‑specific system prompts. 【F:runtime/problem_solving_session.py†L211-L252】【F:llm/prompts/prompts.py†L1-L146】
4. **Peer review and refinement** ensure that each solver can incorporate critiques before final selection. 【F:runtime/problem_solving_session.py†L254-L400】【F:runtime/contexts/solver_agent_context.py†L146-L204】
5. **Final judging** chooses a single winner based on structured evidence from all stages. 【F:runtime/problem_solving_session.py†L402-L401】【F:runtime/contexts/judge_agent_context.py†L40-L169】
6. **Auditable outputs** are persisted to Firestore and JSON files for evaluation and reproducibility. 【F:runtime/problem_solving_session.py†L78-L401】【F:data/persistence/firestore_manager.py†L1-L46】

## 6. How to Run the Project

### Prerequisites

- Python 3.10+
- Access keys for LLM providers:
  - `OPENAI_API_KEY`
  - `GOOGLE_API_KEY`
  - (Optional) `DEEPSEEK_API_KEY`
- Firestore service account JSON referenced by `FIREBASE_CREDENTIALS` (path relative to project root). 【F:data/persistence/firestore_client.py†L1-L42】

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set environment variables (example):

```bash
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
export FIREBASE_CREDENTIALS=path/to/service-account.json
```

### Run the pipeline

```bash
python main.py
```

The run loads problems from `data/datasets/problems.json`, processes them with the multi‑stage pipeline, and writes results to Firestore and local JSON outputs under `data/output/`. 【F:config.py†L17-L21】【F:runtime/problem_solving_session.py†L96-L401】

### Evaluation outputs

- **Firestore**: Collections contain runs, solutions, reviews, refinements, and final judgements. 【F:data/persistence/firestore_manager.py†L1-L46】
- **Local artifacts**: JSON snapshots per stage stored in `data/output/` subfolders for offline inspection. 【F:config.py†L17-L21】【F:runtime/problem_solving_session.py†L96-L401】
- **Metrics dump**: Use `data/persistence/scripts/generate_metrics.py` to dump collection counts for sanity checks. 【F:data/persistence/scripts/generate_metrics.py†L1-L67】
