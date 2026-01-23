import asyncio
from pathlib import Path
from config import *
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

        # -----------------
        # DeepSeek
        # -----------------
        LLMAgentConfig(
            provider="deepseek",
            llm_id="deepseek-chat",
            model="deepseek-chat",
            temperature=0.3,
            top_p=0.9,
        ),
        LLMAgentConfig(
            provider="deepseek",
            llm_id="deepseek-reasoner",
            model="deepseek-reasoner",
            temperature=0.5,
            top_p=0.95,
        ),
    ]


async def main():
    app = ProblemSolvingApp(
        problems_path=Path(PROBLEMS_PATH),
        agent_configs=create_llm_configs(),
        problems_skip=PROBLEMS_SKIP,
        problems_take=PROBLEMS_TAKE,
        output_dir=Path(DEFAULT_OUTPUT_DIR),
    )

    await app.run(
        timeout_sec=DEFAULT_TIMEOUT_SEC,
        log_interval_sec=LOG_INTERVAL_SEC,
    )


if __name__ == "__main__":
    asyncio.run(main())
