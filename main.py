import asyncio
from pipeline.reasoning_pipeline import ReasoningPipeline


def main():
    pipeline = ReasoningPipeline(
        problems_path="data/problems.json",
        problem_start=0,
        problem_count=1
    )
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()
