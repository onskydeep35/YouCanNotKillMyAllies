import asyncio
from pathlib import Path

from runtime.app import ProblemSolvingApp
from schemas.dataclass.agent_config import LLMAgentConfig
from llm.prompts.prompts import *

def create_llm_configs():
    """
    2 OpenAI + 2 Gemini agents with different reasoning styles.
    """
    return [
        # -----------------
        # OpenAI
        # -----------------
        LLMAgentConfig(
            provider="openai",
            llm_id="gpt-4.1",
            model="gpt-4.1",
            temperature=0.2,
            top_p=0.85,
        ),
        LLMAgentConfig(
            provider="openai",
            llm_id="gpt-5-mini",
            model="gpt-5-mini",
            # temperature=0.7,
            # top_p=0.95,
        ),

        # -----------------
        # Gemini
        # -----------------
        LLMAgentConfig(
            provider="gemini",
            llm_id="gemini-3-pro",
            model="gemini-3-pro-preview",
            temperature=0.3,
            top_p=0.9,
        ),
        LLMAgentConfig(
            provider="gemini",
            llm_id="gemini-3-flash",
            model="gemini-3-flash-preview",
            temperature=0.8,
            top_p=0.95,
        ),
    ]


async def main():
    print(build_solver_system_prompt(category="cat"))

    app = ProblemSolvingApp(
        problems_path=Path("data/datasets/problems.json"),
        agent_configs=create_llm_configs(),
        problems_skip=6,
        problems_take=1,
        output_dir=Path("data/output"),
    )

    await app.run(
        timeout_sec=2000,
        log_interval_sec=10,
    )


if __name__ == "__main__":
    asyncio.run(main())
