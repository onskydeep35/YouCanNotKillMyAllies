import asyncio
from pathlib import Path

from pipeline.debating_pipeline import DebatingPipeline

PROBLEMS_PATH = "data/datasets/problems.json"
PROBLEMS_SKIP = 3
PROBLEMS_TAKE = 1

async def main_async():
    problems_path = Path(PROBLEMS_PATH)
    if not problems_path.exists():
        raise FileNotFoundError(f"Problems file not found: {problems_path}")

    pipeline = DebatingPipeline(
        problems_path=str(problems_path),
        problems_skip=PROBLEMS_SKIP,
        problems_take=PROBLEMS_TAKE,
    )

    await pipeline.run()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
