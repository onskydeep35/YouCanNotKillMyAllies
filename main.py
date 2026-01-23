import asyncio
from pathlib import Path
from config import *
from runtime.app import ProblemSolvingApp
from schemas.dataclass.agent_config import LLMAgentConfig

def create_llm_configs():
    """
    2 OpenAI + 2 Gemini agents with different reasoning styles.
    """
    return [
        # -----------------
        # OpenAI
        # -----------------

        LLMAgentConfig(
            provider="gemini",
            llm_id="gemini-3-pro-1",
            model="gemini-3-pro-preview",
            temperature=0.6,
            top_p=0.9,
        ),
        LLMAgentConfig(
            provider="gemini",
            llm_id="gemini-3-flash-1",
            model="gemini-3-flash-preview",
            temperature=0.3,
            top_p=0.95,
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
        # # DeepSeek
        # # -----------------
        # LLMAgentConfig(
        #     provider="deepseek",
        #     llm_id="deepseek-chat",
        #     model="deepseek-chat",
        #     temperature=0.3,
        #     top_p=0.9,
        # ),
        # LLMAgentConfig(
        #     provider="deepseek",
        #     llm_id="deepseek-reasoner",
        #     model="deepseek-reasoner",
        #     temperature=0.5,
        #     top_p=0.95,
        # ),
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
        max_concurrent_sessions=7,
    )


if __name__ == "__main__":
    asyncio.run(main())
