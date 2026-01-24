import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

from data.persistence.firestore_client import get_firestore_client
from data.persistence.firestore_manager import (
    FirestoreManager,
    FINAL_JUDGEMENTS,
)


FINAL_JUDGEMENTS_DIR = Path("C:/Users/v-aoniani/projects/youCanNotKillMyAllies/data/output/final_judgements")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_ids(filename: str) -> Dict[str, str]:
    """
    Expected filename patterns (pick ONE and be consistent):
      - run_<runId>__problem_<problemId>.json
      - <runId>__<problemId>.json
    """
    name = filename.replace(".json", "")
    parts = name.replace("run_", "").replace("problem_", "").split("__")

    if len(parts) == 2:
        return {
            "run_id": parts[0],
            "problem_id": parts[1],
        }

    raise ValueError(f"Cannot extract ids from filename: {filename}")


async def main() -> None:
    load_dotenv()

    db = get_firestore_client()
    firestore = FirestoreManager(db)

    files = sorted(FINAL_JUDGEMENTS_DIR.glob("*.json"))

    if not files:
        print("No final judgement files found.")
        return

    written = 0
    skipped = 0

    for path in files:
        try:
            data = load_json(path)

            # -------------------------------
            # Recover foreign keys
            # -------------------------------
            if "problem_id" not in data or "run_id" not in data:
                ids = extract_ids(path.name)
                data["problem_id"] = data.get("problem_id") or ids["problem_id"]
                data["run_id"] = data.get("run_id") or ids["run_id"]

            # Optional but recommended
            if "judgement_id" not in data:
                data["judgement_id"] = path.stem

            # -------------------------------
            # Hard validation (fail fast)
            # -------------------------------
            required = ["problem_id", "run_id", "answer", "confidence"]
            for r in required:
                if r not in data or data[r] is None:
                    raise ValueError(f"Missing required field '{r}'")

            # -------------------------------
            # Write to Firestore (FIXED)
            # -------------------------------
            await firestore.write(
                collection=FINAL_JUDGEMENTS,
                document=data,
                document_id=data["judgement_id"]
            )

            written += 1

        except Exception as e:
            skipped += 1
            print(f"[SKIP] {path.name}: {e}")

    print(f"Rehydration done. Written={written}, Skipped={skipped}")


if __name__ == "__main__":
    asyncio.run(main())
