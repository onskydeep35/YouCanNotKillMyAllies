from pathlib import Path
import os
import json
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore

# ---------- Load env ----------
load_dotenv()

cred_env = os.getenv("FIREBASE_CREDENTIALS")
if not cred_env:
    raise RuntimeError("FIREBASE_CREDENTIALS not set")

# ---------- Resolve absolute path ----------
BASE_DIR = Path(__file__).resolve().parents[2]

cred_path = BASE_DIR / cred_env

if not cred_path.exists():
    raise FileNotFoundError(f"Firebase credentials not found at {cred_path}")

# ---------- Init Firebase ----------
cred = credentials.Certificate(str(cred_path))
firebase_admin.initialize_app(cred)

db = firestore.client()

# ---------- Load problems.json ----------
with open("../../datasets/problems.json", "r", encoding="utf-8") as f:
    problems = json.load(f)

# ---------- Upload to Firestore ----------
collection_ref = db.collection("Problems")

for problem in problems:
    problem_id = problem["id"]  # use stable ID
    collection_ref.document(problem_id).set(problem)

print(f"✅ Uploaded {len(problems)} problems to Firestore")
