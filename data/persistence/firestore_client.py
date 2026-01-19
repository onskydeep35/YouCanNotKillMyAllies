from pathlib import Path
import os

import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv


_firestore_db = None


def get_firestore_client():
    """
    Returns a singleton Firestore clients.
    Safe to call multiple times.
    """
    global _firestore_db

    if _firestore_db is not None:
        return _firestore_db

    load_dotenv()

    cred_path = os.getenv("FIREBASE_CREDENTIALS")
    if not cred_path:
        raise RuntimeError("FIREBASE_CREDENTIALS env var not set")

    # Resolve path relative to project root
    base_dir = Path(__file__).resolve().parents[2]
    cred_file = base_dir / cred_path

    if not cred_file.exists():
        raise FileNotFoundError(f"Firebase credentials not found: {cred_file}")

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(cred_file))
        firebase_admin.initialize_app(cred)

    _firestore_db = firestore.client()
    return _firestore_db